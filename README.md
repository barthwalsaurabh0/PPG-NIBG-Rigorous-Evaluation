# Non-Invasive Blood Glucose Prediction Using PPG Under Rigorous Evaluation

Code for the paper *Non-Invasive Blood Glucose Prediction Using PPG Under
Rigorous Evaluation*.

We benchmark 13 deep-learning models that estimate CGM glucose from the raw
wrist PPG (BVP) signal of the [PhysioCGM](https://doi.org/10.6084/m9.figshare.28136294)
dataset (10 participants). Models are trained per participant on a
chronological (time-wise) split, with a 30-minute buffer between train,
validation and test to prevent temporal leakage. We also compare this with a
random split (leaky) and leave-one-subject-out, and test how sensitive the
result is to the buffer length and the quality-control thresholds.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Converts the raw PhysioCGM download into parquet files (run once). **Set `RAW_DIR` here.** |
| `helper_functions.py` | Data matching, quality control, splits, training and metrics. **Set `DATA_DIR` and `WORK_DIR` here.** |
| `models.py` | The 13 model architectures and their learning rates (`MODEL_DB`). |
| `run_experiments.py` | Main benchmark: all models, time-wise split (Table 3). |
| `run_mamba_splits.py` | Mamba under random and leave-one-subject-out splits (Table 4). |
| `run_buffer_sensitivity.py` | Buffer of 0, 5, 15, 30, 60 min (Table 7a). |
| `compute_qc_stats.py` | QC sensitivity, step 1: QC values for every segment. |
| `build_qc_datasets.py` | QC sensitivity, step 2: datasets at other thresholds. |
| `run_qc_sensitivity.py` | QC sensitivity, step 3: train on each dataset (Table 7b). |

## Setup

Tested with Python 3.11, PyTorch 2.5.1 and CUDA on Linux.

```bash
pip install -r requirements.txt
pip install mamba-ssm==2.2.4
```

`mamba-ssm` needs an NVIDIA GPU (compute capability 7.0 or newer). Without it,
the other 12 models still run and Mamba is skipped. Mamba is needed for
`run_mamba_splits.py` and the sensitivity scripts. If the install fails, see the
[mamba-ssm install notes](https://github.com/state-spaces/mamba).

Check that all models build:

```bash
python models.py
```

## Data

Download PhysioCGM from https://doi.org/10.6084/m9.figshare.28136294 and
extract each `<subject>_raw.zip` into its own `<subject>_raw` folder.
The zips use LZMA compression, so use 7-Zip (plain `unzip` may fail):
`7z x c1s01_raw.zip -oRAW_DIR/c1s01_raw`

```
RAW_DIR/
  c1s01_raw/
    cgm.csv
    e4/<recording>/BVP.csv, ACC.csv, ...
  c1s02_raw/
  ...
```

Set `RAW_DIR` at the top of `prepare_data.py` and the two paths at the top of
`helper_functions.py` (below), then run:

```bash
python prepare_data.py
```

This writes one folder per participant (`c1s01` ... `c1s05`, `c2s01` ... `c2s05`):

```
DATA_DIR/
  c1s01/
    df_bvp.parquet   columns: datetime, bvp                 (64 Hz)
    df_acc.parquet   columns: datetime, acc_x, acc_y, acc_z (32 Hz)
    df_cgm.parquet   columns: datetime, glucose             (mg/dL)
  c1s02/
  ...
```

Paths at the top of `helper_functions.py`:

```python
DATA_DIR = Path("~/data/PhysioCGM").expanduser()   # data above
WORK_DIR = Path("~/PCGM_analysis").expanduser()    # caches and outputs
```

## Reproducing the paper

After `prepare_data.py`, run the scripts in this order:

```bash
python run_experiments.py          # Table 3 (also builds the dataset cache)
python run_mamba_splits.py         # Table 4
python run_buffer_sensitivity.py   # Table 7a
python compute_qc_stats.py         # Table 7b, step 1
python build_qc_datasets.py        # Table 7b, step 2
python run_qc_sensitivity.py       # Table 7b, step 3
```

`run_experiments.py` must run first. It builds the matched, quality-controlled
dataset in `WORK_DIR/data/`, which the other scripts reuse. Building it the
first time takes a while.

### Outputs

- Metric tables (`metrics_*.parquet`, `summary_*.csv`) next to the scripts and in `sensitivity/`.
- Per-run model checkpoints, predictions and HTML plots in `WORK_DIR/result*/`.

Each row of a metrics file is one (participant, split, model) run, with MAE,
MARD, RMSE, R², Pearson r, Clarke Error Grid zones, and the same metrics for a
baseline that always predicts the training-set mean (`mean_*` columns).

### Notes

- Seeds are fixed per participant. Mamba's GPU kernels are not fully
  deterministic, so its numbers can differ slightly between runs.
- The random split is a deliberate leakage example, not a valid protocol.

## Citation

If you use this code, please cite:

```bibtex
@article{barthwal2026ppg,
  title   = {Non-Invasive Blood Glucose Prediction Using PPG Under Rigorous Evaluation},
  author  = {Saurabh Barthwal and Aman Bhardwaj and Venugopalan Y. Vishnu and Rahul Garg},
  journal = {Procedia Computer Science},
  year    = {2026},
  note    = {5th International Conference on Machine Learning and Data Engineering (ICMLDE 2026). To appear}
}
```

Please also cite the PhysioCGM dataset.
