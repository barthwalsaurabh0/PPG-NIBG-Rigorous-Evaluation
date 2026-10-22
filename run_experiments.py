#!/usr/bin/env python
"""
Main benchmark (Table 3).

Trains and tests all 13 models for each subject with the time-wise split.
The first run also builds the dataset cache.
"""

from pathlib import Path

import torch

from helper_functions import run_experiment, WORK_DIR
from models import MODEL_DB, HAS_MAMBA

# ACC is loaded only for the motion QC check; the models read BVP only.
sensors_info = {
    "BVP": {"value_cols": ["bvp"], "freq_hz": 64},
    "ACC": {"value_cols": ["acc_x", "acc_y", "acc_z"], "freq_hz": 32},
}

fragment = [(3 * 60, 30)]  # (window_size, segment_size)
split_types = ["time"]
time_split_gap = 30 * 60

all_subjects = ["c1s01", "c1s02", "c1s03", "c1s04", "c1s05",
                "c2s01", "c2s02", "c2s03", "c2s04", "c2s05"]

REPO = Path(__file__).resolve().parent
processed_dataset_base_path = WORK_DIR / "data"
experiments_dir_root = WORK_DIR / "result"

epochs = 50
early_stop_patience = 10
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print(f"Device      : {device}")
    print(f"Subjects    : {len(all_subjects)} -> {all_subjects}")
    print(f"Models      : {len(MODEL_DB)} -> {[m['name'] for m in MODEL_DB]}")
    print(f"mamba-ssm   : {HAS_MAMBA}")
    print(f"Total runs  : {len(MODEL_DB) * len(all_subjects) * len(split_types)}\n")

    results_df = run_experiment(
        fragment=fragment,
        processed_dataset_base_path=processed_dataset_base_path,
        all_subjects=all_subjects,
        sensors_info=sensors_info,
        split_types=split_types,
        time_split_gap=time_split_gap,
        model_db=MODEL_DB,
        experiments_dir_root=experiments_dir_root,
        device=device,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        lr=lr,
        output_parquet=REPO / "metrics_all_pcgm.parquet",
    )

    results_df["family"] = results_df["model"].map({m["name"]: m["family"] for m in MODEL_DB})
    results_df["lr"] = results_df["model"].map({m["name"]: m["lr"] for m in MODEL_DB})

    print("\n\n" + "=" * 78)
    print("Per-model summary (mean over subjects)")
    print("=" * 78)
    summary = (
        results_df.groupby(["family", "model"])
        .agg(
            params=("trainable_params", "max"),
            lr=("lr", "max"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mard=("mard", "mean"),
            r2=("r2", "mean"),
            r=("r", "mean"),
            mean_mae=("mean_mae", "mean"),
            ceg_A=("CEG_A", "mean"),
            ceg_B=("CEG_B", "mean"),
            failures=("error", lambda s: s.notna().sum()),
        )
        .sort_values("mae")
    )
    # Skill vs the baseline that always predicts the training mean:
    # > 0 means the model beat it.
    summary["skill_vs_mean_%"] = (
        (summary["mean_mae"] - summary["mae"]) / summary["mean_mae"] * 100
    )
    print(summary.to_string(float_format=lambda v: f"{v:,.3f}"))

    results_df.to_parquet(REPO / "metrics_all_pcgm.parquet", index=False)
    summary.to_csv(REPO / "summary_pcgm.csv")
    print("\nWrote metrics_all_pcgm.parquet and summary_pcgm.csv")
