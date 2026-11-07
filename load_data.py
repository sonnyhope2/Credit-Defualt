"""
load_data.py
Load one Freddie Mac SFLLD sample vintage, name the columns, recode the
"not available" placeholders, and find when (if ever) each loan first defaults.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Edit this if your folder is different. Expects DATA_ROOT/sample_YYYY/sample_orig_YYYY.txt
DATA_ROOT = Path(r"C:\Users\sonny\OneDrive\Documents\Coding\Credit Defualt")

# Column order taken from the SFLLD General User Guide (Release 48). Files have no header row.
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
assert len(ORIG_COLS) == 31 and len(PERF_COLS) == 35

# Placeholder codes the guide uses for "Not Available" (these are NOT real values)
NUMERIC_NA = {
    "credit_score": 9999, "mi_pct": 999, "n_units": 99, "cltv": 999, "dti": 999,
    "ltv": 999, "n_borrowers": 99, "vantage_score": 9999,
}
CATEGORICAL_NA = {
    "first_time_buyer": "9", "occupancy": "9", "channel": "9",
    "loan_purpose": "9", "property_type": "99", "valuation_method": "7",
}

# The only performance columns needed to define default
PERF_USE = ["loan_id", "period", "delinq_status", "zero_balance_code"]

# Zero balance codes that are credit events: third party sale, short sale/charge-off, REO disposition
DEFAULT_ZB_CODES = [2, 3, 9]


def load_orig(year, root=DATA_ROOT):
    path = Path(root) / f"sample_{year}" / f"sample_orig_{year}.txt"
    orig = pd.read_csv(path, sep="|", header=None, names=ORIG_COLS,
                       dtype={"loan_id": str}, low_memory=False)
    for col, code in NUMERIC_NA.items():
        orig[col] = orig[col].mask(orig[col] == code)
    for col, code in CATEGORICAL_NA.items():
        orig[col] = orig[col].mask(orig[col].astype(str) == code)
    orig["vintage"] = year
    return orig


def load_perf(year, root=DATA_ROOT):
    path = Path(root) / f"sample_{year}" / f"sample_perf_{year}.txt"
    return pd.read_csv(path, sep="|", header=None, names=PERF_COLS, usecols=PERF_USE,
                       dtype={"loan_id": str, "delinq_status": str}, low_memory=False)


def add_first_default(orig, perf):
    """Add first_default_mob: months on book at the loan's first default event (NaN = never)."""
    p = perf.merge(orig[["loan_id", "first_payment_date"]], on="loan_id")

    # Months on book from the dates themselves (matches the guide's Loan Age for unmodified loans)
    p["mob"] = ((p["period"] // 100 - p["first_payment_date"] // 100) * 12
                + (p["period"] % 100 - p["first_payment_date"] % 100) + 1)

    dq = pd.to_numeric(p["delinq_status"], errors="coerce")  # 'RA' and 'XX' become NaN
    event = (
        (dq >= 3)                                   # 90+ days delinquent
        | (p["delinq_status"] == "RA")              # REO acquisition
        | p["zero_balance_code"].isin(DEFAULT_ZB_CODES)
    )
    first_mob = p.loc[event].groupby("loan_id")["mob"].min()

    out = orig.copy()
    out["first_default_mob"] = out["loan_id"].map(first_mob)
    return out


if __name__ == "__main__":
    year = 2005
    orig, perf = load_orig(year), load_perf(year)
    df = add_first_default(orig, perf)

    print("loans:", len(df), "| performance rows:", len(perf))
    print("\nDelinquency status counts:\n", perf["delinq_status"].value_counts().head(8))
    print("\nZero balance codes:\n", perf["zero_balance_code"].value_counts(dropna=False))
    print("\nCumulative default rate by months on book:")
    for w in (12, 24, 36, 48, 60):
        print(f"  within {w:>2} months: {(df['first_default_mob'] <= w).mean():.2%}")
    print(f"  ever (to end of data): {df['first_default_mob'].notna().mean():.2%}")
    print("\nShare missing after recoding:\n",
          df[["credit_score", "dti", "ltv", "cltv", "mi_pct"]].isna().mean().round(3))
