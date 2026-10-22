"""
Convert the raw PhysioCGM download into the parquet files used by this code.

Run once before the other scripts. Set RAW_DIR below to the folder holding the
extracted <subject>_raw folders. Output goes to DATA_DIR (set in
helper_functions.py).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from helper_functions import DATA_DIR

# Folder with the extracted c1s01_raw, c1s02_raw, ... folders.
RAW_DIR = Path("~/data/PhysioCGM_raw").expanduser()


def read_empatica_file(file_path, column_names):
    """Read an Empatica E4 CSV and add a local-time datetime column."""
    with open(file_path, "r") as f:
        start_time = float(f.readline().split(",")[0].strip())
        fs = float(f.readline().split(",")[0].strip())
    recording_df = pd.read_csv(file_path, names=column_names, skiprows=2)

    n = len(recording_df)
    time_index = pd.to_datetime(start_time + np.arange(n) / fs, unit="s", utc=True)
    # the CGM uses local time, so convert to Texas (Central) time
    time_index = time_index.tz_convert("America/Chicago")

    recording_df.insert(0, "datetime", time_index)
    return recording_df


def load_df_physiocgm(subject, db_dir=RAW_DIR):
    """Load all BVP and ACC recordings and the CGM file of one subject."""
    ppg_base_dir = os.path.join(db_dir, subject + "_raw", "e4")
    cgm_file = os.path.join(db_dir, subject + "_raw", "cgm.csv")

    recordings = sorted(os.listdir(ppg_base_dir))

    all_acc = []
    all_bvp = []
    for recording in tqdm(recordings):
        acc_path = os.path.join(ppg_base_dir, recording, "ACC.csv")
        bvp_path = os.path.join(ppg_base_dir, recording, "BVP.csv")

        if os.path.exists(acc_path):
            all_acc.append(read_empatica_file(acc_path, column_names=["acc_x", "acc_y", "acc_z"]))
        else:
            print("ACC not found for:", recording)

        if os.path.exists(bvp_path):
            all_bvp.append(read_empatica_file(bvp_path, column_names=["bvp"]))
        else:
            print("BVP not found for:", recording)

    df_acc = pd.concat(all_acc, ignore_index=True)
    df_bvp = pd.concat(all_bvp, ignore_index=True)
    df_cgm = pd.read_csv(cgm_file)
    return df_bvp, df_acc, df_cgm


def preprocess_columns_physiocgm(df_bvp, df_acc, df_cgm):
    """Clean the CGM table and make all timestamps timezone-naive local time."""
    df_cgm = df_cgm[df_cgm["Event Type"] == "EGV"]
    df_cgm = df_cgm.rename(columns={"Timestamp (YYYY-MM-DDThh:mm:ss)": "datetime",
                                    "Glucose Value (mg/dL)": "glucose"})
    df_cgm = df_cgm[["datetime", "glucose"]].copy()

    df_cgm["datetime"] = pd.to_datetime(df_cgm["datetime"])
    df_cgm["glucose"] = pd.to_numeric(df_cgm["glucose"], errors="coerce")
    df_cgm = df_cgm.dropna(subset=["datetime", "glucose"])
    df_cgm["glucose"] = df_cgm["glucose"].astype(int)

    # two overlapping sensor sessions interleave contradictory readings; drop them
    gap = df_cgm["datetime"].diff().dt.total_seconds()
    df_cgm = df_cgm[~((gap < 150) | (gap.shift(-1) < 150))]
    df_cgm = df_cgm.reset_index(drop=True)

    # drop readings with an ambiguous time (daylight saving change)
    df_cgm["datetime"] = df_cgm["datetime"].dt.tz_localize("America/Chicago", ambiguous="NaT")
    df_cgm = df_cgm[df_cgm["datetime"].notna()]

    df_bvp["datetime"] = df_bvp["datetime"].dt.tz_localize(None)
    df_cgm["datetime"] = df_cgm["datetime"].dt.tz_localize(None)
    df_acc["datetime"] = df_acc["datetime"].dt.tz_localize(None)
    return df_bvp, df_acc, df_cgm


if __name__ == "__main__":
    subjects = sorted(p.name[:-len("_raw")] for p in RAW_DIR.glob("*_raw"))
    for subject in subjects:
        print(subject)
        df_bvp, df_acc, df_cgm = load_df_physiocgm(subject)
        df_bvp, df_acc, df_cgm = preprocess_columns_physiocgm(df_bvp, df_acc, df_cgm)

        save_dir = DATA_DIR / subject
        save_dir.mkdir(parents=True, exist_ok=True)
        df_bvp.to_parquet(save_dir / "df_bvp.parquet", index=False)
        df_acc.to_parquet(save_dir / "df_acc.parquet", index=False)
        df_cgm.to_parquet(save_dir / "df_cgm.parquet", index=False)
