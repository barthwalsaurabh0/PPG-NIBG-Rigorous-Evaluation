"""
The 13 deep-learning models compared in the paper.

Every model takes the raw PPG waveform as input and predicts a glucose value.
Mamba is included only if the mamba-ssm package is installed.

Run `python models.py` to check that all models work.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEGMENT_SAMPLES = 1920  # 30 s @ 64 Hz
BVP_HZ = 64


# --------------------------------------------------------------------------------------
# Feature extraction + normalisation (raw single-channel BVP)
# --------------------------------------------------------------------------------------

def get_features_channel_ppg(dataset):
    """Raw PPG waveform as model input."""
    arr_2d = np.vstack(dataset["BVP_bvp"].values)
    return np.expand_dims(arr_2d, axis=1)


def norm_X_channel_function(train_dataset_X, validation_dataset_X, test_dataset_X):
    """Normalise inputs using training-set statistics."""
    mean_X = np.mean(train_dataset_X, axis=(0, 2), keepdims=True)
    std_X = np.std(train_dataset_X, axis=(0, 2), keepdims=True)
    std_X[std_X == 0] = 1.0

    return (
        (train_dataset_X - mean_X) / std_X,
        (validation_dataset_X - mean_X) / std_X,
        (test_dataset_X - mean_X) / std_X,
    )


# --------------------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------------------

class RegressionHead(nn.Module):
    """Final layers that output the glucose value."""

    def __init__(self, in_features, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        w = x.mean(dim=-1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return x * w.unsqueeze(-1)


# --------------------------------------------------------------------------------------
# 1. MLP
# --------------------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_channels=1, seq_len=SEGMENT_SAMPLES,
                 hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = [nn.Flatten()]
        prev = in_channels * seq_len
        for h in hidden:
            layers += [nn.Dropout(dropout), nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers += [nn.Dropout(dropout), nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------------------
# 2. TSMixer
# --------------------------------------------------------------------------------------

class MixerBlock(nn.Module):
    def __init__(self, n_tokens, d_model, token_expansion=2, channel_expansion=2, dropout=0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, n_tokens * token_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_tokens * token_expansion, n_tokens),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * channel_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * channel_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # (B, T, D)
        y = self.norm1(x).transpose(1, 2)          # (B, D, T)
        x = x + self.token_mlp(y).transpose(1, 2)
        x = x + self.channel_mlp(self.norm2(x))
        return x


class TSMixer(nn.Module):
    def __init__(self, in_channels=1, seq_len=SEGMENT_SAMPLES,
                 patch_len=64, d_model=96, depth=4, dropout=0.15):
        super().__init__()
        self.patch_len = patch_len
        n_tokens = seq_len // patch_len            # 1920 / 64 = 30 tokens (1 s each)
        self.embed = nn.Linear(in_channels * patch_len, d_model)
        self.blocks = nn.ModuleList(
            [MixerBlock(n_tokens, d_model, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = RegressionHead(d_model, dropout=dropout)

    def forward(self, x):                          # (B, C, L)
        b, c, l = x.shape
        x = x.reshape(b, c, l // self.patch_len, self.patch_len)
        x = x.permute(0, 2, 1, 3).reshape(b, l // self.patch_len, -1)
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x).mean(dim=1))


# --------------------------------------------------------------------------------------
# 3. FCN
# --------------------------------------------------------------------------------------

class FCN(nn.Module):
    def __init__(self, in_channels=1, filters=(128, 256, 128), kernels=(8, 5, 3)):
        super().__init__()
        blocks = []
        prev = in_channels
        for f, k in zip(filters, kernels):
            blocks += [
                nn.Conv1d(prev, f, kernel_size=k, padding="same", bias=False),
                nn.BatchNorm1d(f),
                nn.ReLU(),
            ]
            prev = f
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev, 1)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------------------
# 4. ResNet
# --------------------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=(8, 5, 3)):
        super().__init__()
        layers = []
        prev = in_channels
        for i, k in enumerate(kernels):
            layers += [
                nn.Conv1d(prev, out_channels, kernel_size=k, padding="same", bias=False),
                nn.BatchNorm1d(out_channels),
            ]
            if i < len(kernels) - 1:
                layers.append(nn.ReLU())
            prev = out_channels
        self.block = nn.Sequential(*layers)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm1d(out_channels))
            if in_channels != out_channels else nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, in_channels=1, filters=(64, 128, 128)):
        super().__init__()
        blocks = []
        prev = in_channels
        for f in filters:
            blocks.append(ResBlock(prev, f))
            prev = f
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev, 1)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


# --------------------------------------------------------------------------------------
# 5. InceptionTime  --  Ismail Fawaz et al. 2020. Multi-scale kernels in parallel; the
#                       reference SOTA CNN for time series. Kernels 39/19/9 @ 64 Hz cover
#                       ~0.6 / 0.3 / 0.14 s, i.e. sub-beat to beat-scale PPG morphology.
# --------------------------------------------------------------------------------------

class InceptionModule(nn.Module):
    def __init__(self, in_channels, nf=32, kernels=(39, 19, 9), bottleneck=32):
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck, 1, bias=False)
            if in_channels > 1 else nn.Identity()
        )
        bn_out = bottleneck if in_channels > 1 else in_channels
        self.convs = nn.ModuleList(
            [nn.Conv1d(bn_out, nf, k, padding="same", bias=False) for k in kernels]
        )
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, nf, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(nf * (len(kernels) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.bottleneck(x)
        out = [conv(z) for conv in self.convs] + [self.maxpool_conv(x)]
        return self.relu(self.bn(torch.cat(out, dim=1)))


class InceptionTime(nn.Module):
    def __init__(self, in_channels=1, nf=32, depth=6, kernels=(39, 19, 9)):
        super().__init__()
        self.depth = depth
        self.modules_list = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        out_ch = nf * (len(kernels) + 1)

        prev = in_channels
        for d in range(depth):
            self.modules_list.append(InceptionModule(prev, nf=nf, kernels=kernels))
            if d % 3 == 2:
                sc_in = in_channels if d == 2 else out_ch
                self.shortcuts.append(
                    nn.Sequential(nn.Conv1d(sc_in, out_ch, 1, bias=False),
                                  nn.BatchNorm1d(out_ch))
                )
            prev = out_ch

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_ch, 1)

    def forward(self, x):
        res = x
        for d in range(self.depth):
            x = self.modules_list[d](x)
            if d % 3 == 2:
                x = self.relu(x + self.shortcuts[d // 3](res))
                res = x
        return self.fc(self.pool(x).flatten(1))


# --------------------------------------------------------------------------------------
# 6. TCN  --  Bai et al. 2018. Dilated residual convolutions. 8 levels of dilation
#             (1..128) with k=7 give a receptive field of ~3060 samples > the whole 30 s
#             segment, so it sees global context with far fewer FLOPs than attention.
# --------------------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.15):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + self.downsample(x))


class TCN(nn.Module):
    def __init__(self, in_channels=1, channels=(64, 64, 64, 64, 64, 64, 64, 64),
                 kernel_size=7, dropout=0.15):
        super().__init__()
        blocks = []
        prev = in_channels
        for i, ch in enumerate(channels):
            blocks.append(TemporalBlock(prev, ch, kernel_size, dilation=2 ** i,
                                        dropout=dropout))
            prev = ch
        self.network = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = RegressionHead(prev, dropout=dropout)

    def forward(self, x):
        return self.head(self.pool(self.network(x)).flatten(1))


# --------------------------------------------------------------------------------------
# 7. BiLSTM  --  pure recurrent model. Fixed average-pool decimation (x8 -> 240 steps)
#                keeps the sequence tractable for BPTT; the decimation is a fixed
#                operator, not a learned feature extractor, so the model stays "pure RNN".
# --------------------------------------------------------------------------------------

class BiLSTM(nn.Module):
    def __init__(self, in_channels=1, pool=8, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.decimate = nn.AvgPool1d(pool)
        self.rnn = nn.LSTM(
            input_size=in_channels, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = RegressionHead(hidden * 4, dropout=dropout)

    def forward(self, x):
        x = self.decimate(x).transpose(1, 2)       # (B, T, C)
        out, _ = self.rnn(x)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(pooled)


# --------------------------------------------------------------------------------------
# 8. CNN-LSTM  --  the most common hybrid in the PPG literature: a strided CNN learns
#                  local pulse morphology, the LSTM models how it evolves over 30 s.
# --------------------------------------------------------------------------------------

class CNN_LSTM(nn.Module):
    def __init__(self, in_channels=1, cnn_filters=(32, 64, 128),
                 hidden=128, layers=2, dropout=0.2):
        super().__init__()
        blocks = []
        prev = in_channels
        for f in cnn_filters:
            blocks += [
                nn.Conv1d(prev, f, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(f),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            prev = f
        self.cnn = nn.Sequential(*blocks)           # 1920 -> 240
        self.rnn = nn.LSTM(
            input_size=prev, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = RegressionHead(hidden * 2, dropout=dropout)

    def forward(self, x):
        x = self.cnn(x).transpose(1, 2)
        out, _ = self.rnn(x)
        return self.head(self.dropout(out.mean(dim=1)))


# --------------------------------------------------------------------------------------
# 9. LSTM-FCN  --  Karim et al. 2018. Two parallel branches (recurrent + convolutional)
#                  concatenated before the head, with squeeze-excite in the FCN branch.
#                  Note: the original dimension-shuffle degenerates for univariate input,
#                  so the LSTM branch reads the x8-decimated waveform instead.
# --------------------------------------------------------------------------------------

class LSTM_FCN(nn.Module):
    def __init__(self, in_channels=1, pool=8, hidden=128, dropout=0.2,
                 filters=(128, 256, 128), kernels=(8, 5, 3)):
        super().__init__()
        self.decimate = nn.AvgPool1d(pool)
        self.rnn = nn.LSTM(in_channels, hidden, batch_first=True)
        self.rnn_dropout = nn.Dropout(dropout)

        conv_blocks = []
        prev = in_channels
        for i, (f, k) in enumerate(zip(filters, kernels)):
            conv_blocks += [
                nn.Conv1d(prev, f, kernel_size=k, padding="same", bias=False),
                nn.BatchNorm1d(f),
                nn.ReLU(),
            ]
            if i < len(filters) - 1:
                conv_blocks.append(SqueezeExcite1D(f))
            prev = f
        self.fcn = nn.Sequential(*conv_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = RegressionHead(hidden + prev, dropout=dropout)

    def forward(self, x):
        r, _ = self.rnn(self.decimate(x).transpose(1, 2))
        r = self.rnn_dropout(r[:, -1, :])
        c = self.pool(self.fcn(x)).flatten(1)
        return self.head(torch.cat([r, c], dim=1))


# --------------------------------------------------------------------------------------
# 10. PatchTransformer  --  PatchTST-style (Nie et al. 2023). The waveform is cut into
#                           overlapping 1 s patches (64 samples, stride 32 -> 59 tokens)
#                           and fed to a standard pre-norm encoder. Patching is a reshape,
#                           not a feature transform.
# --------------------------------------------------------------------------------------

class PatchTransformer(nn.Module):
    def __init__(self, in_channels=1, seq_len=SEGMENT_SAMPLES, patch_len=64, stride=32,
                 d_model=128, n_heads=8, depth=3, ff_mult=4, dropout=0.2):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        n_tokens = (seq_len - patch_len) // stride + 1

        self.embed = nn.Linear(in_channels * patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = RegressionHead(d_model, dropout=dropout)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B,C,T,P)
        b, c, t, p = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * p)
        x = self.drop(self.embed(x) + self.pos)
        x = self.encoder(x)
        return self.head(self.norm(x).mean(dim=1))


# --------------------------------------------------------------------------------------
# 11. ConvTransformer  --  convolutional tokeniser + self-attention. Separates "how are
#                          tokens built" from "how are they mixed": same encoder as above,
#                          but tokens come from a learned strided CNN instead of patching.
# --------------------------------------------------------------------------------------

class ConvTransformer(nn.Module):
    def __init__(self, in_channels=1, d_model=128, n_heads=8, depth=3,
                 ff_mult=4, dropout=0.2, max_tokens=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model), nn.ReLU(),
        )                                            # 1920 -> 240
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = RegressionHead(d_model, dropout=dropout)

    def forward(self, x):
        x = self.stem(x).transpose(1, 2)
        x = self.drop(x + self.pos[:, : x.size(1)])
        x = self.encoder(x)
        return self.head(self.norm(x).mean(dim=1))


# --------------------------------------------------------------------------------------
# 12. S4D  --  Gu et al. 2022, diagonal structured state space. Linear-time global
#              context via an FFT convolution with a learned SSM kernel. Pure PyTorch,
#              no custom CUDA kernels.
# --------------------------------------------------------------------------------------

class S4DKernel(nn.Module):
    def __init__(self, d_model, n_state=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        h, n = d_model, n_state // 2
        log_dt = torch.rand(h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.C = nn.Parameter(torch.randn(h, n, 2) / math.sqrt(2))
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(h, n)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n).repeat(h, 1).float())

    def forward(self, seq_len):
        dt = torch.exp(self.log_dt).unsqueeze(-1)                    # (H, 1)
        C = torch.view_as_complex(self.C)                            # (H, N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag           # (H, N)
        dtA = A * dt                                                 # (H, N)
        K = dtA.unsqueeze(-1) * torch.arange(seq_len, device=A.device)
        C = C * (torch.exp(dtA) - 1.0) / A
        return 2 * torch.einsum("hn,hnl->hl", C, torch.exp(K)).real  # (H, L)


class S4DBlock(nn.Module):
    def __init__(self, d_model, n_state=64, dropout=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.kernel = S4DKernel(d_model, n_state=n_state)
        self.D = nn.Parameter(torch.randn(d_model))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Conv1d(d_model, 2 * d_model, 1), nn.GLU(dim=-2))

    def forward(self, u):                                            # (B, H, L)
        residual = u
        u = self.norm(u.transpose(1, 2)).transpose(1, 2)
        seq_len = u.size(-1)
        k = self.kernel(seq_len)
        k_f = torch.fft.rfft(k, n=2 * seq_len)
        u_f = torch.fft.rfft(u, n=2 * seq_len)
        y = torch.fft.irfft(u_f * k_f, n=2 * seq_len)[..., :seq_len]
        y = y + u * self.D.unsqueeze(-1)
        y = self.out(self.dropout(self.act(y)))
        return residual + self.dropout(y)


class S4D(nn.Module):
    def __init__(self, in_channels=1, d_model=128, depth=4, n_state=64, dropout=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )                                            # 1920 -> 480
        self.blocks = nn.ModuleList(
            [S4DBlock(d_model, n_state=n_state, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = RegressionHead(d_model, dropout=dropout)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x.transpose(1, 2)).mean(dim=1))


# --------------------------------------------------------------------------------------
# 13. Mamba  --  Gu & Dao 2023, selective state space. Optional: only registered if the
#                mamba-ssm CUDA package imports successfully.
# --------------------------------------------------------------------------------------

try:
    from mamba_ssm import Mamba as _MambaMixer

    HAS_MAMBA = True
except Exception:  # pragma: no cover - depends on local CUDA toolchain
    _MambaMixer = None
    HAS_MAMBA = False


class Mamba1D(nn.Module):
    def __init__(self, in_channels=1, d_model=128, depth=4, d_state=16,
                 d_conv=4, expand=2, dropout=0.15):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba-ssm is not available in this environment")
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )                                            # 1920 -> 480
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.mixers = nn.ModuleList([
            _MambaMixer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = RegressionHead(d_model, dropout=dropout)

    def forward(self, x):
        x = self.stem(x).transpose(1, 2)             # (B, T, D)
        for norm, mixer in zip(self.norms, self.mixers):
            x = x + self.dropout(mixer(norm(x)))
        return self.head(self.norm(x).mean(dim=1))


# --------------------------------------------------------------------------------------
# Model registry consumed by helper_functions.run_experiment
# --------------------------------------------------------------------------------------

LR_CONV = 1e-3  # MLP / CNN / RNN / hybrid families: the standard Adam setting
LR_ATTN = 3e-4   # Transformers and SSMs are unstable at 1e-3 without warmup


def _entry(name, model_class, model_params, family, lr=LR_CONV):
    return {
        "name": name,
        "family": family,
        "model_class": model_class,
        "model_params": model_params,
        "get_features_function": get_features_channel_ppg,
        "norm_X_function": norm_X_channel_function,
        "loss_function": nn.MSELoss(),
        "lr": lr,  # consumed by helper_functions.run_experiment (falls back to global lr)
    }


MODEL_DB = [
    # --- MLP / Mixer -----------------------------------------------------------------
    _entry("MLP", MLP, {"in_channels": 1, "seq_len": SEGMENT_SAMPLES}, "MLP"),
    _entry("TSMixer", TSMixer, {"in_channels": 1, "seq_len": SEGMENT_SAMPLES}, "MLP"),

    # --- Convolutional ---------------------------------------------------------------
    _entry("FCN", FCN, {"in_channels": 1}, "CNN"),
    _entry("ResNet", ResNet, {"in_channels": 1}, "CNN"),
    _entry("InceptionTime", InceptionTime, {"in_channels": 1}, "CNN"),
    _entry("TCN", TCN, {"in_channels": 1}, "CNN"),

    # --- Recurrent -------------------------------------------------------------------
    _entry("BiLSTM", BiLSTM, {"in_channels": 1}, "RNN"),

    # --- Hybrid ----------------------------------------------------------------------
    _entry("CNN_LSTM", CNN_LSTM, {"in_channels": 1}, "Hybrid"),
    _entry("LSTM_FCN", LSTM_FCN, {"in_channels": 1}, "Hybrid"),

    # --- Attention -------------------------------------------------------------------
    _entry("PatchTransformer", PatchTransformer,
           {"in_channels": 1, "seq_len": SEGMENT_SAMPLES}, "Transformer", lr=LR_ATTN),
    _entry("ConvTransformer", ConvTransformer, {"in_channels": 1}, "Transformer", lr=LR_ATTN),

    # --- State space -----------------------------------------------------------------
    _entry("S4D", S4D, {"in_channels": 1}, "SSM", lr=LR_ATTN),
]

if HAS_MAMBA:
    MODEL_DB.append(_entry("Mamba", Mamba1D, {"in_channels": 1}, "SSM", lr=LR_ATTN))


def count_parameters(entry):
    """Number of trainable parameters of a model."""
    model = entry["model_class"](**entry["model_params"])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Mamba's selective-scan kernel is CUDA-only, so run the shape check on GPU if present.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 1, SEGMENT_SAMPLES, device=device)

    print(f"device: {device} | mamba-ssm available: {HAS_MAMBA}\n")
    print(f"{'model':<20} {'family':<12} {'params':>12}  output")
    print("-" * 62)
    for entry in MODEL_DB:
        model = entry["model_class"](**entry["model_params"]).to(device).eval()
        with torch.no_grad():
            y = model(x)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{entry['name']:<20} {entry['family']:<12} {n:>12,}  {tuple(y.shape)}")
