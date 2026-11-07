"""Read raw SFLLD sample files: name columns, recode placeholders, find each loan's first default."""
from pathlib import Path

import pandas as pd

from config import RAW_DIR

# Column order from the SFLLD General User Guide (Release 48). The files have no header row.
ORIG_COLS = [
    "credit_score", "first_payment_date", "first_time_buyer", "maturity_date", "msa",
    "mi_pct", "n_units", "occupancy", "cltv", "dti", "orig_upb", "ltv", "int_rate",
    "channel", "prepay_penalty", "amort_type", "state", "property_type", "postal_code",
    "loan_id", "loan_purpose", "orig_term", "n_borrowers", "seller", "super_conforming",
    "pre_harp_id", "special_program", "harp", "valuation_method", "io_indicator",
    "vantage_score",
]
PERF_COLS = [
    "loan_id", "period", "current_upb", "delinq_status", "loan_age", "months_to_maturity",
    "defect_settlement_date", "mod_flag", "zero_balance_code", "zero_balance_date",
    "current_int_rate", "non_int_bearing_upb", "ddlpi", "mi_recoveries", "net_sale_proceeds",
    "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_costs",
    "taxes_insurance", "misc_expenses", "actual_loss", "cum_mod_costs",
    "rate_step_indicator", "payment_deferral", "eltv", "zero_balance_removal_upb",
    "delinquent_accrued_interest", "disaster_delinquency", "borrower_assistance_plan",
    "current_period_mod_costs", "interest_bearing_upb", "mi_cancel_indicator", "servicer",
    "bankruptcy_cramdown_costs",
]

# Placeholder codes the guide uses for "Not Available". They are not real values.
NUMERIC_NA = {
    "credit_score": 9999, "mi_pct": 999, "n_units": 99, "cltv": 999, "dti": 999,
    "ltv": 999, "n_borrowers": 99, "vantage_score": 9999,
}
CATEGORICAL_NA = {
    "first_time_buyer": "9", "occupancy": "9", "channel": "9",
    "loan_purpose": "9", "property_type": "99", "valuation_method": "7",
}

# Only these performance columns are needed to define default (keeps memory low)
PERF_USE = ["loan_id", "period", "delinq_status", "zero_balance_code"]

# Zero balance codes that are credit events: 02 third party sale, 03 short sale or
# charge-off, 09 REO disposition. 01 (prepaid), 15/16 (loan sales) and 96 (defect) are not.
DEFAULT_ZB_CODES = [2, 3, 9]


def _check_n_columns(path, expected):
    """Fail loudly if the layout changes: with names= pandas would otherwise misalign silently."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        n = f.readline().count("|") + 1
    if n != expected:
        raise ValueError(f"{path.name}: expected {expected} columns but found {n}. "
                         "Check the layout against the user guide.")


def available_years(root=RAW_DIR):
    """Vintage years that have both files in root/sample_YYYY/."""
    years = []
    for d in sorted(Path(root).glob("sample_*")):
        y = d.name.split("_")[-1]
        if d.is_dir() and y.isdigit() and (d / f"sample_orig_{y}.txt").exists() \
                and (d / f"sample_perf_{y}.txt").exists():
            years.append(int(y))
    return years


def load_orig(year, root=RAW_DIR):
    path = Path(root) / f"sample_{year}" / f"sample_orig_{year}.txt"
    _check_n_columns(path, len(ORIG_COLS))
    orig = pd.read_csv(path, sep="|", header=None, names=ORIG_COLS,
                       dtype={"loan_id": str}, low_memory=False)
    for col, code in NUMERIC_NA.items():
        orig[col] = orig[col].mask(orig[col] == code)
    for col, code in CATEGORICAL_NA.items():
        orig[col] = orig[col].mask(orig[col].astype(str) == code)
    orig["vintage"] = year
    return orig


def load_perf(year, root=RAW_DIR):
    path = Path(root) / f"sample_{year}" / f"sample_perf_{year}.txt"
    _check_n_columns(path, len(PERF_COLS))
    return pd.read_csv(path, sep="|", header=None, names=PERF_COLS, usecols=PERF_USE,
                       dtype={"loan_id": str, "delinq_status": str}, low_memory=False)


def add_first_default(orig, perf):
    """Add first_default_mob: months on book at the loan's first default event (NaN = never)."""
    p = perf.merge(orig[["loan_id", "first_payment_date"]], on="loan_id")

    # Months on book from the dates. Not the loan_age column: the guide says loan age
    # restarts after a modification, which would make late defaults look early.
    p["mob"] = ((p["period"] // 100 - p["first_payment_date"] // 100) * 12
                + (p["period"] % 100 - p["first_payment_date"] % 100) + 1)

    dq = pd.to_numeric(p["delinq_status"], errors="coerce")  # 'RA' and 'XX' become NaN
    event = (
        (dq >= 3)                                    # 90+ days delinquent
        | (p["delinq_status"] == "RA")               # REO acquisition
        | p["zero_balance_code"].isin(DEFAULT_ZB_CODES)
    )
    first_mob = p.loc[event].groupby("loan_id")["mob"].min()

    out = orig.copy()
    out["first_default_mob"] = out["loan_id"].map(first_mob)
    return out
