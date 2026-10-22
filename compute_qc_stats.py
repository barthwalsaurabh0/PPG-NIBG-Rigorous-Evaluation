#!/usr/bin/env python
"""
QC sensitivity, step 1 of 3.

Computes the motion and pulsatility values behind the QC checks for every
segment, so other thresholds can be tried without recomputing them.
Run run_experiments.py first.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import butter, filtfilt

from helper_functions import WORK_DIR

RAW = WORK_DIR / "data" / "dataset_30_180.parquet"
CHECKS = WORK_DIR / "data" / "pre_processed_checks_info_30_180.parquet"
OUT = Path(__file__).resolve().parent / "sensitivity" / "qc_stats_continuous.parquet"

BVP_FS, ACC_FS = 64, 32
BATCH = 1024

# thresholds used in the paper, needed here only for the check against stored flags
ACC_THRESHOLD = 1.4
AC_THRESHOLD = 0.8

# max_autocorrelation(): lag window for 35-220 bpm at 64 Hz
MIN_LAG = int((60 / 220) * BVP_FS)      # 17
MAX_LAG = int((60 / 35) * BVP_FS)       # 109

B_ACC, A_ACC = butter(4, 0.5 / (ACC_FS / 2), btype="highpass")
B_BVP, A_BVP = butter(2, [0.5 / (BVP_FS / 2), 4.0 / (BVP_FS / 2)], btype="band")


def acc_mean_magnitude(x, y, z):
    """Motion value for many segments at once (same as check_acc_magnitude)."""
    xh = filtfilt(B_ACC, A_ACC, x, axis=1)
    yh = filtfilt(B_ACC, A_ACC, y, axis=1)
    zh = filtfilt(B_ACC, A_ACC, z, axis=1)
    return np.sqrt(xh**2 + yh**2 + zh**2).mean(axis=1)


def ppg_max_autocorr(bvp):
    """Pulsatility value for many segments at once (same as check_ppg_ac)."""
    f = filtfilt(B_BVP, A_BVP, bvp, axis=1)
    f = f - f.mean(axis=1, keepdims=True)
    n = f.shape[1]
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    F = np.fft.rfft(f, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :n]
    ac = ac / ac[:, :1]
    return ac[:, MIN_LAG:MAX_LAG].max(axis=1)


def main():
    pf = pq.ParquetFile(RAW)
    total = pf.metadata.num_rows
    print(f"pre-QC segments: {total:,}")

    cols = ["glucose_index", "segment_index", "glucose_time", "glucose_value",
            "subject", "BVP_bvp", "ACC_acc_x", "ACC_acc_y", "ACC_acc_z"]

    rows, done = [], 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        d = batch.to_pandas()
        bvp = np.vstack(d["BVP_bvp"].values).astype(np.float64)
        ax = np.vstack(d["ACC_acc_x"].values).astype(np.float64)
        ay = np.vstack(d["ACC_acc_y"].values).astype(np.float64)
        az = np.vstack(d["ACC_acc_z"].values).astype(np.float64)

        rows.append(pd.DataFrame({
            "subject": d["subject"].values,
            "glucose_index": d["glucose_index"].values,
            "segment_index": d["segment_index"].values,
            "glucose_time": d["glucose_time"].values,
            "glucose_value": d["glucose_value"].values,
            "acc_mean_hp_magnitude": acc_mean_magnitude(ax, ay, az),
            "ppg_max_autocorr": ppg_max_autocorr(bvp),
        }))
        done += len(d)
        if done % (BATCH * 20) == 0 or done == total:
            print(f"  {done:,}/{total:,}", flush=True)

    stats = pd.concat(rows, ignore_index=True)

    # ---- check against the pass/fail flags saved by the main pipeline --------
    checks = pd.read_parquet(CHECKS)
    assert len(checks) == len(stats), (len(checks), len(stats))
    mine_acc = stats["acc_mean_hp_magnitude"].values <= ACC_THRESHOLD
    mine_ppg = stats["ppg_max_autocorr"].values >= AC_THRESHOLD
    d_acc = int((mine_acc != checks["check_acc_magnitude"].values).sum())
    d_ppg = int((mine_ppg != checks["check_ppg_ac"].values).sum())

    print("\nvalidation against pre_processed_checks_info (must be 0):")
    print(f"  motion-check disagreements      : {d_acc}")
    print(f"  pulsatility-check disagreements : {d_ppg}")
    print(f"  pass rates: motion {100*mine_acc.mean():.2f}% "
          f"(stored {100*checks['check_acc_magnitude'].mean():.2f}%), "
          f"pulsatility {100*mine_ppg.mean():.2f}% "
          f"(stored {100*checks['check_ppg_ac'].mean():.2f}%), "
          f"both {100*(mine_acc & mine_ppg).mean():.2f}%")
    if d_acc or d_ppg:
        print("\nWARNING: vectorised statistics do not reproduce the stored checks.")
        print("Inspect before using this file for the sensitivity analysis.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(stats):,} rows)")


if __name__ == "__main__":
    main()
