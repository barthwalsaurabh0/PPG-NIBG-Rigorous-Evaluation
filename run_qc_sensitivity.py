#!/usr/bin/env python
"""
QC sensitivity, step 3 of 3 (Table 7b).

Trains Mamba on each dataset from build_qc_datasets.py.
Results are saved in sensitivity/.
"""

from pathlib import Path

import pandas as pd
import torch

from helper_functions import run_experiment, WORK_DIR
from models import MODEL_DB

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "sensitivity"
QC_ROOT = WORK_DIR / "qc_variants"

sensors_info = {
    "BVP": {"value_cols": ["bvp"], "freq_hz": 64},
    "ACC": {"value_cols": ["acc_x", "acc_y", "acc_z"], "freq_hz": 32},
}

fragment = [(3 * 60, 30)]
VARIANTS = ["ac_strict", "ac_loose", "acc_strict", "acc_loose"]
TIME_SPLIT_GAP = 30 * 60

all_subjects = ["c1s01", "c1s02", "c1s03", "c1s04", "c1s05",
                "c2s01", "c2s02", "c2s03", "c2s04", "c2s05"]

epochs, early_stop_patience, lr = 50, 10, 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "Mamba"
model_db = [m for m in MODEL_DB if m["name"] == MODEL_NAME]
assert model_db, (f"{MODEL_NAME} not in MODEL_DB. mamba-ssm needs a GPU of compute "
                  "capability 7.0+; it will not run on Pascal (GTX 10xx).")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [v for v in VARIANTS if not (QC_ROOT / v / "pre_processed_dataset_30_180.parquet").exists()]
    assert not missing, f"run build_qc_datasets.py first; missing variants: {missing}"

    print(f"Device   : {device}")
    print(f"Model    : {MODEL_NAME} (lr={model_db[0]['lr']})")
    print(f"Variants : {VARIANTS}")
    print(f"Runs     : {len(VARIANTS) * len(all_subjects)}\n")

    frames = []
    for tag in VARIANTS:
        print("\n" + "=" * 92)
        print(f"QC variant: {tag}")
        print("=" * 92)
        df = run_experiment(
            fragment=fragment,
            processed_dataset_base_path=QC_ROOT / tag,
            all_subjects=all_subjects,
            sensors_info=sensors_info,
            split_types=["time"],
            time_split_gap=TIME_SPLIT_GAP,
            model_db=model_db,
            experiments_dir_root=QC_ROOT / tag / "result",
            device=device,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            lr=lr,
            output_parquet=str(OUT_DIR / f"metrics_qc_{tag}.parquet"),
        )
        df["qc_variant"] = tag
        frames.append(df)

    # default thresholds: reuse the main benchmark result
    base = pd.read_parquet(REPO / "metrics_all_pcgm.parquet")
    base = base[(base["model"] == MODEL_NAME) & (base["split_type"] == "time")].copy()
    base = base.drop(columns=[c for c in ("family", "lr") if c in base.columns])
    base["qc_variant"] = "default"
    frames.append(base)

    merged = pd.concat(frames, ignore_index=True)
    merged["skill_vs_mean_%"] = (
        (merged["mean_mae"] - merged["mae"]) / merged["mean_mae"] * 100
    )
    merged.to_parquet(OUT_DIR / "metrics_qc_sensitivity.parquet", index=False)

    summary = (merged.groupby("qc_variant")
               .agg(n=("subject", "size"), mae=("mae", "mean"), rmse=("rmse", "mean"),
                    mard=("mard", "mean"), r2=("r2", "mean"), r=("r", "mean"),
                    baseline=("mean_mae", "mean"), skill=("skill_vs_mean_%", "mean"),
                    n_pos_r2=("r2", lambda s: (s > 0).sum()),
                    failures=("error", lambda s: s.notna().sum()))
               .reindex(["default"] + VARIANTS))
    summary.to_csv(OUT_DIR / "summary_qc_sensitivity.csv")

    print("\n\n" + "=" * 92)
    print(f"{MODEL_NAME}, time-wise split, by QC operating point (mean over 10 subjects)")
    print("=" * 92)
    print(summary.to_string(float_format=lambda v: f"{v:,.3f}"))
    print(f"\nwrote {OUT_DIR}/metrics_qc_sensitivity.parquet")
