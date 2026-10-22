"""
Data loading, quality control, data splits, training and evaluation.

Set DATA_DIR and WORK_DIR below before running any script.
"""

import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm

import hashlib
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from concurrent.futures import ProcessPoolExecutor
from functools import partial


import clarke_error_grid as ceg
import traceback

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---- Paths: edit these two lines for your machine ----------------------------------
# DATA_DIR: PhysioCGM data, one folder per subject (see README for the layout).
# WORK_DIR: dataset caches and training outputs are written here.
DATA_DIR = Path("~/data/PhysioCGM").expanduser()
WORK_DIR = Path("~/PCGM_analysis").expanduser()


# PhysioCGM dataset
def get_subject_raw_dataset_PCGM(subject, sensors_info):
    """Load the PPG, accelerometer and glucose data of one subject."""
    sensor_files = {"BVP": "df_bvp.parquet", "ACC": "df_acc.parquet"}

    raw_dataset = dict()
    for sensor in sensors_info:
        raw_dataset[sensor] = pd.read_parquet(DATA_DIR / str(subject) / sensor_files[sensor])

    cgm_file = DATA_DIR / subject / "df_cgm.parquet"
    raw_dataset["GLUC"] = pd.read_parquet(cgm_file)

    return raw_dataset


def get_subject_matched_dataset(raw_dataset, segment_length, sensors_info, window_size=120):
    """Pair each glucose reading with the sensor segments recorded just before it."""
    segments_n = window_size // segment_length
    tolerance = pd.Timedelta(1, unit="seconds")

    if window_size>600 or segments_n<=0:
        raise ValueError("Invalid window or segment size")

    segment_duration = pd.Timedelta(segment_length, unit="seconds")
    other_sensors = sorted(raw_dataset.keys() - {"GLUC"})


    # Init
    combined_dataset = {"glucose_index":[], "segment_index": [], "glucose_time": [], "glucose_value": [] }
    for sensor in other_sensors:
        combined_dataset[f"{sensor}_datetime"]= []
        for col in sensors_info[sensor]["value_cols"]:
            combined_dataset[f"{sensor}_{col}"]= []


    if not raw_dataset["GLUC"]["datetime"].is_monotonic_increasing:
        raise ValueError("Glucose DateTime Not Monotonic")


    for sensor in other_sensors:
        if "freq_hz" not in sensors_info[sensor]:
            continue
        if not raw_dataset[sensor]["datetime"].is_monotonic_increasing:
            raise ValueError(f"Sensor DateTime Not Monotonic: {sensor}")


    for glucose_index, row in tqdm(raw_dataset["GLUC"].iterrows(), desc="Datetime matching", total=len(raw_dataset["GLUC"])):
        glucose_value = row["glucose"]
        glucose_time = row["datetime"]


        for i in range(segments_n):
            segment_end = glucose_time - (i * segment_duration)

            sensors_segment_df = {}
            for sensor in other_sensors:

                if "freq_hz" not in sensors_info[sensor]:
                    continue

                sensor_df = raw_dataset[sensor]
                sensor_times = sensor_df["datetime"]


                sensor_segment_sample_size = round(segment_length * sensors_info[sensor]["freq_hz"])

                sensor_segment_end_idx = sensor_times.searchsorted(segment_end)
                if sensor_segment_end_idx >= len(sensor_times):
                    sensor_segment_end_idx = len(sensor_times) - 1


                sensor_segment_end_time = sensor_times.iloc[sensor_segment_end_idx]
                sensor_segment_start_idx = sensor_segment_end_idx - sensor_segment_sample_size  # so dataset contains constant size segment for all

                if abs(sensor_segment_end_time - segment_end) > tolerance:
                    continue

                if sensor_segment_start_idx<0:
                    continue

                sensor_segment_start_time = sensor_times.iloc[sensor_segment_start_idx]


                if abs(abs(sensor_segment_end_time - sensor_segment_start_time) - segment_duration) > tolerance:
                    continue

                sensor_segment_df = sensor_df.iloc[sensor_segment_start_idx:sensor_segment_end_idx].reset_index(drop=True)

                sensors_segment_df[sensor] = sensor_segment_df

            combined_dataset["glucose_index"].append(glucose_index)
            combined_dataset["segment_index"].append(i)
            combined_dataset["glucose_time"].append(glucose_time)
            combined_dataset["glucose_value"].append(glucose_value)

            for sensor in other_sensors:

                if sensors_segment_df.get(sensor, None) is not None:
                    for column in sensors_segment_df[sensor].columns:
                        combined_dataset[f"{sensor}_{column}"].append(sensors_segment_df[sensor][column].values)
                else:
                    combined_dataset[f"{sensor}_datetime"].append(None)
                    if "value_cols" in sensors_info[sensor]:
                        for col in sensors_info[sensor]["value_cols"]:
                            combined_dataset[f"{sensor}_{col}"].append(None)


    return combined_dataset


def _build_one_subject(subject, segment_length, sensors_info, window_size, dataset_subject_function=get_subject_raw_dataset_PCGM):
    """Build the matched dataset for one subject."""
    print(f"Creating Dataset for {subject}")
    raw_dataset_subject = dataset_subject_function(subject, sensors_info)

    cd = get_subject_matched_dataset(raw_dataset_subject, segment_length, sensors_info, window_size)
    cdf = pd.DataFrame(cd)

    # keep sensor value columns only (drop timestamps and IBI)
    non_datetime_cols = [
        col for col in cdf.columns
        if not col.endswith("datetime") and "IBI" not in col
    ]

    scdf = cdf[non_datetime_cols]
    scdf = scdf.dropna().reset_index(drop=True)
    scdf["subject"] = str(subject)  # string id

    return scdf


def get_matched_dataset(subjects, segment_length, sensors_info, window_size=600, max_workers=16):
    """Build the matched dataset for all subjects in parallel."""
    worker = partial(_build_one_subject,
                     segment_length=segment_length,
                     sensors_info=sensors_info,
                     window_size=window_size)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        dataset_list = list(ex.map(worker, subjects))   # map preserves input order

    dataset = pd.concat(dataset_list, ignore_index=True)

    return dataset


def max_autocorrelation(data, min_hr = 35, max_hr = 220, fs = 64):
    """Strongest autocorrelation in the heart-rate range. High means a clear pulse."""
    max_point = int((60/min_hr)*fs)
    min_point = int((60/max_hr)*fs)

    data = data - np.mean(data) # Mean-center the data

    # Autocorrelation
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[autocorr.size // 2:] # Keep only positive lags
    autocorr = autocorr / autocorr[0] # Norm

    max_ac = autocorr[min_point:max_point].max()
    return max_ac


def check_acc_magnitude(row, cutoff=0.5, order=4, acc_fs=32, acc_mean_magnitude_threshold=1.4):
    """Motion check: True if the segment has little movement."""
    b, a = butter(order, cutoff / (acc_fs / 2), btype="highpass")

    x_hp = filtfilt(b, a, row["ACC_acc_x"])
    y_hp = filtfilt(b, a, row["ACC_acc_y"])
    z_hp = filtfilt(b, a, row["ACC_acc_z"])

    hp_mag = np.sqrt(x_hp**2 + y_hp**2 + z_hp**2)
    mean_magnitude = np.mean(hp_mag)

    if (mean_magnitude <= acc_mean_magnitude_threshold):
        return True
    else:
        return False


def filter_ppg(data, bvp_fs, lowcut=0.5, highcut=4.0, order=2):
    """Band-pass filter for PPG."""
    nyq = 0.5 * bvp_fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def check_ppg_ac(row, bvp_fs=64, max_ac_threshold=0.8):
    """Pulsatility check: True if the PPG shows a clear pulse."""
    ppg = row["BVP_bvp"]
    filtered_ppg = filter_ppg(ppg, bvp_fs=bvp_fs)
    max_ac = max_autocorrelation(filtered_ppg, fs=bvp_fs)

    if max_ac >= max_ac_threshold:
        return True
    else:
        return False


def pre_process_dataset(dataset):
    """Apply the QC checks and keep only the segments that pass."""
    # Format [(check_name, check_function),]
    checks = [("check_acc_magnitude", check_acc_magnitude),
              ("check_ppg_ac", check_ppg_ac),]

    checks_rows = []

    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        checks_row = []
        for _, check_function in checks:
            checks_row.append(check_function(row)) # True if passes, False if fails
        checks_rows.append(checks_row)

    columns = [check_name for check_name, _ in checks]
    checks_df = pd.DataFrame(checks_rows, columns=columns, index=dataset.index)

    good_indices = checks_df.index[checks_df.all(axis=1)]
    dataset = dataset.loc[good_indices].reset_index(drop=True)

    return dataset, checks_df


def get_pre_processed_dataset(processed_dataset_base_path, all_subjects, segment_length, window_size, sensors_info):
    """Load the QC-filtered dataset, building and caching it the first time."""
    pre_processed_dataset_file_path = processed_dataset_base_path / Path(f"pre_processed_dataset_{segment_length}_{window_size}.parquet")
    checks_info_file_path = processed_dataset_base_path / Path(f"pre_processed_checks_info_{segment_length}_{window_size}.parquet")

    if pre_processed_dataset_file_path.exists():
        print(f"Reading Pre Processed Cached: {pre_processed_dataset_file_path}")
        dataset = pd.read_parquet(pre_processed_dataset_file_path)
        checks_df = pd.read_parquet(checks_info_file_path)
    else:
        processed_dataset_base_path.mkdir(parents=True, exist_ok=True)
        dataset_file_path = processed_dataset_base_path / Path(f"dataset_{segment_length}_{window_size}.parquet")
        if dataset_file_path.exists():
            print(f"Reading Cached: {dataset_file_path}")
            dataset = pd.read_parquet(dataset_file_path)
        else:
            dataset = get_matched_dataset(all_subjects, segment_length, sensors_info, window_size)
            print(f"Caching: {dataset_file_path}")
            dataset.to_parquet(dataset_file_path, compression=None, index=False)

        dataset, checks_df = pre_process_dataset(dataset) # remove segments with motion artifacts and other potential checks
        print(f"Caching Pre Processed: {pre_processed_dataset_file_path}")
        dataset.to_parquet(pre_processed_dataset_file_path, compression=None, index=False)
        checks_df.to_parquet(checks_info_file_path)


    # checks stats
    print("\n\nQC pass stats:")
    print("-" * 30)
    checks_stats = checks_df.mean()
    for check_name, ratio in checks_stats.items():
        print(f"{check_name}: {ratio*100:.2f}%")

    print(f"Overall: {(checks_df.all(axis=1).mean())*100:.2f}%")
    print("-" * 30)

    return dataset


class model_dataset(Dataset):
    """PyTorch Dataset for (X, y) arrays."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_features_raw_ppg(dataset, ppg_column="BVP_bvp"):
    """Raw PPG waveform as model input."""
    arr_2d = np.vstack(dataset[ppg_column].values)
    return np.expand_dims(arr_2d, axis=1)


def get_split_dataset(dataset, split_type, info, get_features_function, norm_X_function, random_seed=42, verbose=True):
    """Split the data into train, validation and test sets, then normalise.

    split_type is "time" (chronological, with a buffer between sets),
    "random" (shuffled; leaks, used only for comparison) or
    "loso" (leave-one-subject-out).
    """
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 1 - (train_ratio + validation_ratio)


    train_subjects = info["train_subjects"]

    if split_type=="loso":
        validation_subjects = info["validation_subjects"]
        test_subjects = info["test_subjects"]

        train_dataset = dataset[dataset["subject"].isin(train_subjects)]
        validation_dataset = dataset[dataset["subject"].isin(validation_subjects)]
        test_dataset = dataset[dataset["subject"].isin(test_subjects)]

        train_features_X = get_features_function(train_dataset)
        validation_features_X = get_features_function(validation_dataset)
        test_features_X = get_features_function(test_dataset)

        train_dataset_X, train_dataset_y = train_features_X, train_dataset["glucose_value"].values.reshape(-1, 1)
        validation_dataset_X, validation_dataset_y = validation_features_X, validation_dataset["glucose_value"].values.reshape(-1, 1)
        test_dataset_X, test_dataset_y = test_features_X, test_dataset["glucose_value"].values.reshape(-1, 1)


    # leaks across splits (see docstring); used only as an ablation
    elif split_type=="random":

        train_dataset = dataset[dataset["subject"].isin(train_subjects)]

        features_X = get_features_function(train_dataset)
        features_y = train_dataset["glucose_value"].values.reshape(-1, 1)

        l = len(features_X)

        train_split_idx = int(train_ratio * l)
        validation_split_idx = int((train_ratio + validation_ratio) * l)


        rng = np.random.default_rng(seed=random_seed)
        indices = rng.permutation(l)
        features_X = features_X[indices]
        features_y = features_y[indices]


        train_dataset_X, train_dataset_y = features_X[:train_split_idx], features_y[:train_split_idx]
        validation_dataset_X, validation_dataset_y = features_X[train_split_idx:validation_split_idx], features_y[train_split_idx:validation_split_idx]
        test_dataset_X, test_dataset_y = features_X[validation_split_idx:], features_y[validation_split_idx:]


    elif split_type=="time":
        # train |<-- gap -->| boundary | validation
        # validation |<-- gap/2 -->| boundary |<-- gap/2 -->| test

        dataset = dataset[dataset["subject"].isin(train_subjects)].reset_index(drop=True)

        if not dataset["glucose_time"].is_monotonic_increasing:
            raise ValueError("glucose_time not monotonic — time split boundaries would be invalid")


        features_X = get_features_function(dataset)
        features_y = dataset["glucose_value"].values.reshape(-1, 1)

        l = len(features_X)

        train_split_idx = int(train_ratio * l)
        validation_split_idx = int((train_ratio + validation_ratio) * l)

        time_split_gap = info.get("time_split_gap", 30*60) # 30 minutes default

        gap = pd.Timedelta(seconds=time_split_gap)
        half_gap = (gap/2)


        new_train_start_time = dataset.iloc[0]["glucose_time"]
        new_train_end_time = dataset.iloc[train_split_idx]["glucose_time"] - gap

        new_train_end_idx = dataset["glucose_time"].searchsorted(
            new_train_end_time,
            side="left"
        )

        new_validation_start_idx = train_split_idx
        new_validation_start_time = dataset.iloc[new_validation_start_idx]["glucose_time"]
        new_validation_end_time = dataset.iloc[validation_split_idx]["glucose_time"] - half_gap

        new_validation_end_idx = dataset["glucose_time"].searchsorted(
            new_validation_end_time,
            side="left"
        )

        new_test_start_time = dataset.iloc[validation_split_idx]["glucose_time"] + half_gap
        new_test_end_time = dataset.iloc[-1]["glucose_time"]

        new_test_start_idx = dataset["glucose_time"].searchsorted(
            new_test_start_time,
            side="left"
        )


        train_dataset_X, train_dataset_y = features_X[:new_train_end_idx], features_y[:new_train_end_idx]
        validation_dataset_X, validation_dataset_y = features_X[new_validation_start_idx:new_validation_end_idx], features_y[new_validation_start_idx:new_validation_end_idx]
        test_dataset_X, test_dataset_y = features_X[new_test_start_idx:], features_y[new_test_start_idx:]

        if verbose:
            print("Time split:")
            print(f"Train start time: {new_train_start_time}")
            print(f"Train end time: {new_train_end_time}")


            print(f"Validation start time: {new_validation_start_time}")
            print(f"Validation end time: {new_validation_end_time}")

            print(f"Test start time: {new_test_start_time}")
            print(f"Test end time: {new_test_end_time}")


    ## Normalization y
    scaler_y = StandardScaler()

    train_dataset_y = scaler_y.fit_transform(train_dataset_y)
    validation_dataset_y = scaler_y.transform(validation_dataset_y)
    test_dataset_y = scaler_y.transform(test_dataset_y)


    train_dataset_X, validation_dataset_X, test_dataset_X = norm_X_function(train_dataset_X, validation_dataset_X, test_dataset_X)
    return train_dataset_X, train_dataset_y, validation_dataset_X, validation_dataset_y, test_dataset_X, test_dataset_y, scaler_y


def get_dataset_loader(train_dataset_X, train_dataset_y, validation_dataset_X, validation_dataset_y, test_dataset_X, test_dataset_y):
    """Create the PyTorch data loaders."""

    ## Tensor Dataset and Loader
    train_dataset = model_dataset(train_dataset_X, train_dataset_y)
    validation_dataset = model_dataset(validation_dataset_X, validation_dataset_y)
    test_dataset = model_dataset(test_dataset_X, test_dataset_y)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, validation_loader, test_loader


def set_random_seeds(random_seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def train_model(model, device, train_loader, validation_loader, epochs=1, early_stop_patience=10, loss_function=nn.MSELoss(), lr=0.001, artifacts_dir=Path()):
    """Train the model with early stopping and return the best one."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_epoch = None

    best_avg_validation_epoch_loss = float('inf')
    patience_counter = 0

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = artifacts_dir / "best_model.pt"

    train_losses = []
    validation_losses = []

    for epoch in range(1, epochs+1):
        train_epoch_loss = 0
        validation_epoch_loss = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch = X_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_train_batch)
            train_loss_batch = loss_function(y_pred, y_train_batch)
            train_loss_batch.backward()

            train_epoch_loss += train_loss_batch.item()

            optimizer.step()

        model.eval()
        with torch.inference_mode():
            for X_validation_batch, y_validation_batch in validation_loader:
                X_validation_batch = X_validation_batch.to(device)
                y_validation_batch = y_validation_batch.to(device)

                y_pred = model(X_validation_batch)
                validation_loss_batch = loss_function(y_pred, y_validation_batch)
                validation_epoch_loss += validation_loss_batch.item()


        avg_train_epoch_loss = train_epoch_loss / len(train_loader)
        avg_validation_epoch_loss = validation_epoch_loss / len(validation_loader)

        scheduler.step(avg_validation_epoch_loss)
        current_lr = optimizer.param_groups[0]["lr"]


        improved = (np.isfinite(avg_validation_epoch_loss) and avg_validation_epoch_loss < best_avg_validation_epoch_loss)

        if not improved:
            patience_counter += 1
        else:
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            best_avg_validation_epoch_loss = avg_validation_epoch_loss


        train_losses.append(avg_train_epoch_loss)
        validation_losses.append(avg_validation_epoch_loss)

        if patience_counter >= early_stop_patience:
            print(f"Early Patience Triggered at Epoch: {epoch:02d}")
            break

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_epoch_loss:.3f} | Validation Loss {avg_validation_epoch_loss:.3f} | LR: {current_lr:.6f}")

    if best_epoch is None:
        print("Warning: validation loss never improved (NaN/diverged) — keeping last-epoch weights")
    else:
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    return model, train_losses, validation_losses, best_epoch


## Metrics
def clarke_error_grid(actual, predicted):
    """Percentage of points in each Clarke Error Grid zone (A-E)."""
    zone = ceg.zone(actual, predicted)
    zone_label = {"A": zone[0], "B": zone[1], "C": zone[2], "D": zone[3], "E": zone[4]}

    return zone_label


def mean_absolute_relative_difference(y_true, y_pred, eps=1e-6):
    """Mean absolute relative difference (MARD), in percent."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true > eps   # avoid divide-by-zero
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mard = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    return mard


def metrics_calculation(actuals, predictions, verbose=True):
    """Compute MAE, MARD, MSE, RMSE, R2 and Pearson r."""
    mae = mean_absolute_error(actuals, predictions)
    mard = mean_absolute_relative_difference(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    r, p_value = pearsonr(actuals, predictions)


    if verbose:
        print("MAE:", mae)
        print("MARD:", mard)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2:", r2)
        print("Pearson r:", r)


    return mae, mard, mse, rmse, r2, r


## Plots

# Predictions and Actuals (Time Series Plot)
def plot_glucose_time_series(actuals, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actuals, mode="lines", name="Actuals"))
    fig.add_trace(go.Scatter(y=predictions, mode="lines", name="Predictions"))
    fig.update_layout(height=600, title="Actuals vs Predictions (Time Series PLot)", xaxis_title="Segment", yaxis_title="Glucose Value")
    return fig


# Predictions and Actuals (Scatter Plot)
def plot_glucose_scatter(actuals, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actuals, y=predictions, mode="markers"))
    fig.update_layout(height=600, width=600, title="Actuals vs Predictions (Scatter Plot)", xaxis_title="Actual", yaxis_title="Prediction")
    return fig


# Predictions and Actuals (CEG Plot)
def plot_glucose_ceg(actuals, predictions):
    fig = ceg.plot(actuals, predictions)
    return fig


def plot_losses(train_losses, validation_losses, test_loss, best_epoch):
    """Plot training and validation loss."""
    epochs = list(range(1, len(train_losses) + 1))

    train_color = "#1f77b4"
    val_color = "#ff7f0e"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=epochs, y=train_losses, line=dict(color=train_color), name="Train Loss")
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=validation_losses, line=dict(color=val_color), name="Validation Loss")
    )

    if best_epoch is not None:
        fig.add_trace(
            go.Scatter(
                x=[best_epoch],
                y=[train_losses[best_epoch-1]],
                mode="markers",
                marker=dict(size=12, color=train_color),
                name="Best Val Epoch's Train Loss"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[best_epoch],
                y=[validation_losses[best_epoch-1]],
                mode="markers",
                marker=dict(size=12, color=val_color),
                name="Best Val Epoch's Val Loss"
            )
        )

    fig.update_layout(
        title=f"Test Loss: {test_loss:.4f}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=600,
    )

    return fig


def evaluate_model(model, device, test_loader, loss_function=nn.MSELoss()):
    """Get the model's predictions on the test set."""
    model.eval()

    actuals = []
    predictions = []

    test_loss = 0

    with torch.inference_mode():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)

            pred_batch = model(X_test_batch)

            actuals.extend(y_test_batch.flatten().tolist())
            predictions.extend(pred_batch.flatten().tolist())


            test_loss_batch = loss_function(pred_batch, y_test_batch)
            test_loss += test_loss_batch.item()

        avg_test_loss = test_loss / len(test_loader)

    return avg_test_loss, actuals, predictions


def run_experiment(
    fragment: list,
    processed_dataset_base_path: Path,
    all_subjects: list,
    sensors_info: dict,
    split_types: list,
    time_split_gap: int,
    model_db: list,
    experiments_dir_root: Path,
    device: torch.device,
    epochs: int = 50,
    early_stop_patience: int = 10,
    lr: float = 1e-3,
    output_parquet: str = "metrics_all.parquet",
):
    """Train and test each model for each subject and split type.

    Returns a table of test metrics (one row per run) and saves it to output_parquet.
    """
    metric_rows = []
    for window_size, segment_length in fragment:
        dataset = get_pre_processed_dataset(processed_dataset_base_path, all_subjects, segment_length, window_size, sensors_info)

        missing = set(all_subjects) - set(dataset["subject"].unique())
        assert not missing, f"cache {processed_dataset_base_path} missing subjects: {sorted(missing)}"

        for split_type in split_types:
            for index, subject in enumerate(all_subjects):
                random_seed = int(hashlib.md5(str(subject).encode()).hexdigest()[:8], 16)

                info = {}
                if split_type == "loso":
                    if len(all_subjects) < 3:
                        print(f"Skipping LOSO, need >=3 subjects, have {len(all_subjects)}")
                        break

                    rng = random.Random(random_seed)
                    info["test_subjects"] = [subject]  # current subject as test subject
                    pool = [i for i in all_subjects if not i in info["test_subjects"]]
                    info["validation_subjects"] = rng.sample(pool, 1)
                    info["train_subjects"] = [i for i in pool if i not in info["validation_subjects"]]

                if split_type == "time":
                    info["train_subjects"] = [subject]
                    info["time_split_gap"] = time_split_gap

                if split_type == "random":
                    info["train_subjects"] = [subject]


                for model_defination in model_db:
                    set_random_seeds(random_seed=random_seed)

                    model_name = model_defination["name"]
                    trainable_params = np.nan
                    model_lr = model_defination.get("lr", lr)  # per-model lr, falls back to global
                    norm_X_function = model_defination["norm_X_function"]
                    get_features_function = model_defination["get_features_function"]
                    loss_function = model_defination["loss_function"]
                    artifacts_dir = experiments_dir_root / f"segment_{segment_length}" / str(subject) / model_name / split_type

                    try:
                        model = model_defination["model_class"](**model_defination["model_params"]).to(device)
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                        print(f"\n\nSubject: {subject} ({index+1:02d}/{len(all_subjects):02d}) | Segment_Length: {segment_length} | Split: {split_type} | Model: {model_name} | LR: {model_lr}")

                        train_dataset_X, train_dataset_y, validation_dataset_X, validation_dataset_y, test_dataset_X, test_dataset_y, scaler_y = get_split_dataset(dataset, split_type=split_type, info=info, get_features_function=get_features_function, norm_X_function=norm_X_function, random_seed=random_seed)
                        train_loader, validation_loader, test_loader = get_dataset_loader(train_dataset_X, train_dataset_y, validation_dataset_X, validation_dataset_y, test_dataset_X, test_dataset_y)

                        model, train_losses, validation_losses, best_epoch = train_model(model, device, train_loader, validation_loader,
                                                                                             epochs=epochs, early_stop_patience=early_stop_patience,
                                                                                             loss_function=loss_function,
                                                                                             lr=model_lr, artifacts_dir=artifacts_dir)


                        avg_test_loss, actuals, predictions = evaluate_model(model, device, test_loader, loss_function=loss_function)


                        actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).reshape(-1)
                        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).reshape(-1)

                        np.save(artifacts_dir / "actuals.npy", actuals)
                        np.save(artifacts_dir / "predictions.npy", predictions)

                        loss_fig = plot_losses(train_losses, validation_losses, test_loss=avg_test_loss, best_epoch=best_epoch)
                        scatter_fig = plot_glucose_scatter(actuals, predictions)
                        series_fig = plot_glucose_time_series(actuals, predictions)

                        loss_fig.write_html(artifacts_dir / "loss_plot.html")
                        scatter_fig.write_html(artifacts_dir / "scatter_plot.html")
                        series_fig.write_html(artifacts_dir / "series_plot.html")

                        mae, mard, mse, rmse, r2, r = metrics_calculation(actuals=actuals, predictions=predictions)
                        ceg_percentage = clarke_error_grid(actuals, predictions)


                        predictions_mean = np.full(shape=len(actuals), fill_value=train_dataset_y.mean())
                        predictions_mean = scaler_y.inverse_transform(np.array(predictions_mean).reshape(-1, 1)).reshape(-1)

                        mean_mae, mean_mard, mean_mse, mean_rmse, mean_r2, mean_r = metrics_calculation(actuals=actuals, predictions=predictions_mean)
                        # mean_r would be nan because of constant

                        metric_rows.append((subject, segment_length, model_name, trainable_params, split_type, mae, mard, mse, rmse, r2, r, mean_mae, mean_mard, mean_mse, mean_rmse, mean_r2, ceg_percentage["A"], ceg_percentage["B"], ceg_percentage["C"], ceg_percentage["D"], ceg_percentage["E"], None))

                    except Exception as e:
                        print(f"Failed: {e}")
                        traceback.print_exc()

                        metric_rows.append((
                            subject, segment_length, model_name, trainable_params, split_type,
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, str(e)
                        ))


    columns = [
        "subject", "segment_length", "model", "trainable_params", "split_type",
        "mae", "mard", "mse", "rmse", "r2", "r",
        "mean_mae", "mean_mard", "mean_mse", "mean_rmse", "mean_r2",
        "CEG_A", "CEG_B", "CEG_C", "CEG_D", "CEG_E", "error"
    ]

    metrics_df = pd.DataFrame(metric_rows, columns=columns)
    metrics_df.to_parquet(output_parquet, index=False)

    return metrics_df
