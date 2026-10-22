#!/usr/bin/env python
"""
QC sensitivity, step 2 of 3.

Builds four datasets, each with one QC threshold made stricter or looser.
Run compute_qc_stats.py first.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from helper_functions import WORK_DIR

REPO = Path(__file__).resolve().parent
RAW = WORK_DIR / "data" / "dataset_30_180.parquet"
STATS = REPO / "sensitivity" / "qc_stats_continuous.parquet"
OUT_ROOT = WORK_DIR / "qc_variants"
BATCH = 2048

# tag -> (min autocorrelation, max mean high-passed ACC magnitude)
VARIANTS = {
    "ac_strict":  (0.9, 1.4),   # pulsatility tightened
    "ac_loose":   (0.7, 1.4),   # pulsatility relaxed
    "acc_strict": (0.8, 1.0),   # motion tightened
    "acc_loose":  (0.8, 2.2),   # motion relaxed
}
DEFAULT = (0.8, 1.4)


def build(tag, ac_thr, acc_thr, stats):
    keep_ppg = stats["ppg_max_autocorr"].values >= ac_thr
    keep_acc = stats["acc_mean_hp_magnitude"].values <= acc_thr
    keep = keep_ppg & keep_acc

    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # QC flags for all 221,450 segments at this variant's thresholds, so the
    # pass rates printed by get_pre_processed_dataset are correct
    pd.DataFrame({"check_acc_magnitude": keep_acc,
                  "check_ppg_ac": keep_ppg}
                 ).to_parquet(out_dir / "pre_processed_checks_info_30_180.parquet")

    pf = pq.ParquetFile(RAW)
    writer, off, written = None, 0, 0
    for batch in pf.iter_batches(batch_size=BATCH):
        n = batch.num_rows
        mask = pa.array(keep[off:off + n])
        sel = pa.Table.from_batches([batch]).filter(mask)
        off += n
        if sel.num_rows == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(out_dir / "pre_processed_dataset_30_180.parquet",
                                      sel.schema, compression=None)
        writer.write_table(sel)
        written += sel.num_rows
    if writer is not None:
        writer.close()

    assert written == int(keep.sum()), (written, int(keep.sum()))
    per = stats.assign(k=keep).groupby("subject")["k"].sum()
    print(f"{tag:7s} ac>={ac_thr} acc<={acc_thr}: {written:,} segments "
          f"({100*keep.mean():.2f}% of {len(keep):,})")
    print(f"         per subject: {dict(per)}")
    return {"tag": tag, "ac_threshold": ac_thr, "acc_threshold": acc_thr,
            "n_segments": written, "retention_pct": 100 * float(keep.mean())}


def main():
    stats = pd.read_parquet(STATS)
    rows = []
    ac, acc = DEFAULT
    n_def = int(((stats["ppg_max_autocorr"] >= ac)
                 & (stats["acc_mean_hp_magnitude"] <= acc)).sum())
    print(f"default ac>={ac} acc<={acc}: {n_def:,} segments (the dataset in the paper)\n")
    rows.append({"tag": "default", "ac_threshold": ac, "acc_threshold": acc,
                 "n_segments": n_def, "retention_pct": 100 * n_def / len(stats)})

    for tag, (a, c) in VARIANTS.items():
        rows.append(build(tag, a, c, stats))

    out = REPO / "sensitivity" / "qc_variants.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"variants under {OUT_ROOT}")


if __name__ == "__main__":
    main()
