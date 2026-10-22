#!/usr/bin/env python
"""
Split comparison (Table 4).

Trains Mamba with the random split and the leave-one-subject-out split.
Run run_experiments.py first (the time-wise results are taken from there).
"""

from pathlib import Path

import pandas as pd
import torch

from helper_functions import run_experiment, WORK_DIR
from models import MODEL_DB

sensors_info = {
    "BVP": {"value_cols": ["bvp"], "freq_hz": 64},
    "ACC": {"value_cols": ["acc_x", "acc_y", "acc_z"], "freq_hz": 32},
}

fragment = [(3 * 60, 30)]
split_types = ["random", "loso"]          # "time" comes from run_experiments.py
time_split_gap = 30 * 60

all_subjects = ["c1s01", "c1s02", "c1s03", "c1s04", "c1s05",
                "c2s01", "c2s02", "c2s03", "c2s04", "c2s05"]

processed_dataset_base_path = WORK_DIR / "data"
experiments_dir_root = WORK_DIR / "result"

epochs = 50
early_stop_patience = 10
lr = 1e-3                                  # overridden per-model by the entry's "lr"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO = Path(__file__).resolve().parent
TIME_RESULTS = REPO / "metrics_all_pcgm.parquet"
OUT_RESULTS = REPO / "metrics_mamba_splits.parquet"
MERGED = REPO / "metrics_mamba_all_splits.parquet"

mamba_db = [m for m in MODEL_DB if m["name"] == "Mamba"]
assert mamba_db, "Mamba not in MODEL_DB. Is mamba-ssm installed (needs a CUDA GPU)?"


if __name__ == "__main__":
    print(f"Device     : {device}")
    print(f"Model      : Mamba (lr={mamba_db[0]['lr']})")
    print(f"Splits     : {split_types}")
    print(f"Total runs : {len(mamba_db) * len(all_subjects) * len(split_types)}")
    print("Note: LOSO trains on ~43k segments per fold (~20x the time-split folds).\n")

    df = run_experiment(
        fragment=fragment,
        processed_dataset_base_path=processed_dataset_base_path,
        all_subjects=all_subjects,
        sensors_info=sensors_info,
        split_types=split_types,
        time_split_gap=time_split_gap,
        model_db=mamba_db,
        experiments_dir_root=experiments_dir_root,
        device=device,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        lr=lr,
        output_parquet=OUT_RESULTS,
    )

    # Add the time-split Mamba rows from the main benchmark.
    time_df = pd.read_parquet(TIME_RESULTS)
    time_mamba = time_df[time_df["model"] == "Mamba"].drop(
        columns=[c for c in ("family", "lr") if c in time_df.columns]
    )
    merged = pd.concat([time_mamba, df], ignore_index=True)
    merged["skill_vs_mean_%"] = (
        (merged["mean_mae"] - merged["mae"]) / merged["mean_mae"] * 100
    )
    merged.to_parquet(MERGED, index=False)

    print("\n\n" + "=" * 92)
    print("Mamba across split protocols (mean over 10 subjects)")
    print("=" * 92)
    summary = (
        merged.groupby("split_type")
        .agg(
            n=("subject", "size"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mard=("mard", "mean"),
            r2=("r2", "mean"),
            r=("r", "mean"),
            mean_mae=("mean_mae", "mean"),
            skill=("skill_vs_mean_%", "mean"),
            ceg_A=("CEG_A", "mean"),
            ceg_B=("CEG_B", "mean"),
            failures=("error", lambda s: s.notna().sum()),
        )
        .reindex(["time", "random", "loso"])
    )
    print(summary.to_string(float_format=lambda v: f"{v:,.3f}"))

    print("\n\nPer-subject MAE by split")
    print("-" * 92)
    pivot = merged.pivot_table(index="subject", columns="split_type", values="mae")
    pivot = pivot.reindex(columns=[c for c in ["time", "random", "loso"]
                                   if c in pivot.columns])
    pivot["baseline"] = merged.groupby("subject")["mean_mae"].max()
    print(pivot.to_string(float_format=lambda v: f"{v:,.2f}"))

    print("\n\nPer-subject R2 by split")
    print("-" * 92)
    pr2 = merged.pivot_table(index="subject", columns="split_type", values="r2")
    pr2 = pr2.reindex(columns=[c for c in ["time", "random", "loso"] if c in pr2.columns])
    print(pr2.to_string(float_format=lambda v: f"{v:,.3f}"))

    summary.to_csv(REPO / "summary_mamba_splits.csv")
    print(f"\nWrote {OUT_RESULTS.name}, {MERGED.name}, summary_mamba_splits.csv")
