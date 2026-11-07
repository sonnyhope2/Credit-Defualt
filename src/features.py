"""Feature engineering, the default label and the out-of-time split (shared by every later step)."""
import json

import numpy as np
import pandas as pd

from config import (IN_TIME_TEST_FRAC, LOANS_PATH, META_PATH, SEED, TEST_YEARS, TRAIN_YEARS,
                    VALID_YEARS, WINDOW)

# Origination-time information only, so nothing here leaks the outcome
NUMERIC = ["credit_score", "dti", "ltv", "mi_pct", "log_upb", "rate_spread"]
BINARY = ["has_second_lien", "multi_borrower", "multi_unit", "harp", "super_conforming"]
CATEGORICAL = ["occupancy", "loan_purpose", "channel", "property_type", "first_time_buyer",
               "state", "term_group"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Directions imposed on the monotone LightGBM (+1: risk rises with the feature, -1: falls).
# Only textbook relationships are constrained; everything else is left free (0).
MONOTONE = {"credit_score": -1, "dti": 1, "ltv": 1, "rate_spread": 1}


def add_months(yyyymm, k):
    """Add k months to YYYYMM integers."""
    total = (yyyymm // 100) * 12 + (yyyymm % 100 - 1) + k
    return (total // 12) * 100 + total % 12 + 1


def build_features(loans):
    df = loans.copy()
    df["log_upb"] = np.log(df["orig_upb"])
    df["term_group"] = np.select([df["orig_term"] <= 180, df["orig_term"] <= 240],
                                 ["15y", "20y"], default="30y")

    # Rate spread at origination: the loan's rate minus the median rate of loans from the
    # same origination quarter and term. Raw rates mostly track the market (6% in 2006, 4% in
    # 2015), which would leak "which year is it" into the model. The spread keeps the part
    # of pricing that reflects the borrower's risk.
    quarter = df["loan_id"].str[1:5]  # e.g. "05Q1" from F05Q1XXXXXXX
    median_rate = df.groupby([quarter, df["term_group"]])["int_rate"].transform("median")
    df["rate_spread"] = df["int_rate"] - median_rate

    # CLTV is only reported when it is at least the LTV, so CLTV > LTV means a second lien
    df["has_second_lien"] = (df["cltv"] > df["ltv"]).astype(int)
    # Before 2018Q2 the borrower count is coded 1 or "2 = more than one", so use a binary flag
    df["multi_borrower"] = (df["n_borrowers"] >= 2).astype(int)
    df["multi_unit"] = (df["n_units"] > 1).astype(int)
    df["harp"] = (df["harp"] == "Y").fillna(False).astype(int)
    df["super_conforming"] = (df["super_conforming"] == "Y").fillna(False).astype(int)

    # Missing categories become their own level rather than being imputed
    for col in CATEGORICAL:
        df[col] = df[col].fillna("missing").astype(str)
    return df


def assign_split(df):
    vintage = df["vintage"].to_numpy()
    split = np.full(len(df), "monitor", dtype=object)       # later vintages: stability checks
    split[vintage < min(TRAIN_YEARS)] = "unused"
    split[np.isin(vintage, TRAIN_YEARS)] = "train"
    split[np.isin(vintage, VALID_YEARS)] = "valid"
    split[np.isin(vintage, TEST_YEARS)] = "test"
    # A random slice of the training vintages, to compare in-time vs out-of-time performance
    rng = np.random.default_rng(SEED)
    split[(split == "train") & (rng.random(len(df)) < IN_TIME_TEST_FRAC)] = "test_in_time"
    return split


def load_modelling_data(window=WINDOW):
    """Loans with features, the 0/1 default label and a split column."""
    loans = pd.read_parquet(LOANS_PATH)
    cutoff = json.loads(META_PATH.read_text())["performance_cutoff"]
    df = build_features(loans)
    df["default"] = (df["first_default_mob"] <= window).astype(int)

    # Keep only loans whose whole window falls before the end of the data. The rule uses the
    # first payment date alone, so it does not depend on what happened to the loan. Loans that
    # prepaid inside the window stay in as non-defaults: they can no longer default.
    window_end = add_months(df["first_payment_date"].astype(int), window - 1)
    df = df[window_end <= cutoff].copy()
    df["split"] = assign_split(df)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_modelling_data()
    summary = df.groupby("split")["default"].agg(loans="size", defaults="sum", rate="mean")
    print(summary.to_string(formatters={"rate": "{:.2%}".format}))
    print("\nMissing share in features:\n", df[FEATURES].isna().mean().round(4).to_string())
    print("\nRate spread by vintage (should be centred near 0):")
    print(df.groupby("vintage")["rate_spread"].median().round(3).to_string())
