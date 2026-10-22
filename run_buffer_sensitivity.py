#!/usr/bin/env python
"""
Buffer sensitivity (Table 7a).

Trains Mamba with different buffer lengths (0 to 60 minutes) between the
train, validation and test sets. Results are saved in sensitivity/.
"""

from pathlib import Path

import pandas as pd
import torch

from helper_functions import run_experiment, WORK_DIR
from models import MODEL_DB

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "sensitivity"

sensors_info = {
    "BVP": {"value_cols": ["bvp"], "freq_hz": 64},
    "ACC": {"value_cols": ["acc_x", "acc_y", "acc_z"], "freq_hz": 32},
}

fragment = [(3 * 60, 30)]
BUFFERS_MIN = [0, 5, 15, 30, 60]

all_subjects = ["c1s01", "c1s02", "c1s03", "c1s04", "c1s05",
                "c2s01", "c2s02", "c2s03", "c2s04", "c2s05"]

processed_dataset_base_path = WORK_DIR / "data"
artifact_root = WORK_DIR / "result_sensitivity"

epochs, early_stop_patience, lr = 50, 10, 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "Mamba"
model_db = [m for m in MODEL_DB if m["name"] == MODEL_NAME]
assert model_db, (f"{MODEL_NAME} not in MODEL_DB. mamba-ssm needs a GPU of compute capability 7.0+")

PUBLISHED_30MIN_MAE = 29.81     # Mamba, time-wise split, Table 3 of the paper


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device  : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    print(f"Model   : {MODEL_NAME} (lr={model_db[0]['lr']})")
    print(f"Buffers : {BUFFERS_MIN} minutes")
    print(f"Runs    : {len(BUFFERS_MIN) * len(all_subjects)}\n")

    frames = []
    for gap_min in BUFFERS_MIN:
        out = OUT_DIR / f"metrics_buffer_{gap_min}min.parquet"
        print("\n" + "=" * 92)
        print(f"buffer = {gap_min} min")
        print("=" * 92)
        df = run_experiment(
            fragment=fragment,
            processed_dataset_base_path=processed_dataset_base_path,
            all_subjects=all_subjects,
            sensors_info=sensors_info,
            split_types=["time"],
            time_split_gap=gap_min * 60,
            model_db=model_db,
            experiments_dir_root=artifact_root / f"result_buffer_{gap_min}min",
            device=device,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            lr=lr,
            output_parquet=str(out),
        )
        df["buffer_min"] = gap_min
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["skill_vs_mean_%"] = (
        (merged["mean_mae"] - merged["mae"]) / merged["mean_mae"] * 100
    )
    merged.to_parquet(OUT_DIR / "metrics_buffer_sensitivity.parquet", index=False)

    summary = (merged.groupby("buffer_min")
               .agg(n=("subject", "size"), mae=("mae", "mean"), rmse=("rmse", "mean"),
                    mard=("mard", "mean"), r2=("r2", "mean"), r=("r", "mean"),
                    baseline=("mean_mae", "mean"), skill=("skill_vs_mean_%", "mean"),
                    n_pos_r2=("r2", lambda s: (s > 0).sum()),
                    failures=("error", lambda s: s.notna().sum())))
    summary.to_csv(OUT_DIR / "summary_buffer_sensitivity.csv")

    print("\n\n" + "=" * 92)
    print(f"{MODEL_NAME}, time-wise split, by buffer duration (mean over 10 subjects)")
    print("=" * 92)
    print(summary.to_string(float_format=lambda v: f"{v:,.3f}"))

    got = float(summary.loc[30, "mae"])
    print(f"\nreproducibility check at 30 min: {got:.2f} mg/dL "
          f"vs {PUBLISHED_30MIN_MAE} published "
          f"({'MATCH' if abs(got - PUBLISHED_30MIN_MAE) < 0.5 else 'DIFFERS -- investigate'})")
    print(f"\nspread across buffers: MAE {summary['mae'].min():.2f} to "
          f"{summary['mae'].max():.2f} mg/dL "
          f"(range {summary['mae'].max() - summary['mae'].min():.2f})")
    print(f"wrote {OUT_DIR}/metrics_buffer_sensitivity.parquet")
