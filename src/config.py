"""Project settings: paths, the default definition window and the out-of-time split."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Folder holding the unzipped sample_YYYY folders. Override with the SFLLD_RAW_DIR
# environment variable if your data lives elsewhere (e.g. a OneDrive folder).
RAW_DIR = Path(os.environ.get(
    "SFLLD_RAW_DIR", r"C:\Users\sonny\OneDrive\Documents\Coding\Credit Defualt\data"))
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

LOANS_PATH = PROCESSED_DIR / "loans.parquet"
META_PATH = PROCESSED_DIR / "meta.json"
PREDICTIONS_PATH = PROCESSED_DIR / "predictions.parquet"

# Label: default within this many months of the first payment date
WINDOW = 24

# Out-of-time design, by origination year (vintage)
TRAIN_YEARS = list(range(2000, 2013))   # 2000-2012: model fitting
VALID_YEARS = [2013, 2014]              # model selection, early stopping, recalibration
TEST_YEARS = [2015, 2016, 2017]         # out-of-time test, untouched until evaluation
COVID_FROM_YEAR = 2018                  # 24-month windows from here overlap COVID forbearance
IN_TIME_TEST_FRAC = 0.20                # random hold-out from the training years

SEED = 42

MODEL_NAMES = ["lr_linear", "lr_binned", "lgbm", "lgbm_monotone"]
MODEL_LABELS = {
    "lr_linear": "Logistic regression",
    "lr_binned": "Logistic regression (binned)",
    "lgbm": "LightGBM",
    "lgbm_monotone": "LightGBM (monotone)",
}
