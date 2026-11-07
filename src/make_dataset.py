"""Step 1: build one row per loan (origination fields + month of first default) for all vintages.

Run:  python src/make_dataset.py
Output: data/processed/loans.parquet and data/processed/meta.json
"""
import json
import time

import pandas as pd

from config import (LOANS_PATH, META_PATH, PROCESSED_DIR, RAW_DIR, TEST_YEARS, TRAIN_YEARS,
                    VALID_YEARS, WINDOW)
from load_data import add_first_default, available_years, load_orig, load_perf

# Origination fields kept for modelling (everything else is unused or only exists for recent years)
KEEP_NUMERIC = ["credit_score", "first_payment_date", "mi_pct", "n_units", "cltv", "dti",
                "orig_upb", "ltv", "int_rate", "orig_term", "n_borrowers"]
KEEP_CATEGORICAL = ["loan_id", "first_time_buyer", "occupancy", "channel", "state",
                    "property_type", "loan_purpose", "super_conforming", "harp"]
KEEP = KEEP_NUMERIC + KEEP_CATEGORICAL + ["vintage", "first_default_mob"]


def main():
    years = available_years()
    if not years:
        raise SystemExit(f"No sample_YYYY folders found in {RAW_DIR}")
    missing = sorted(set(TRAIN_YEARS + VALID_YEARS + TEST_YEARS) - set(years))
    if missing:
        print("WARNING: vintages needed for the train/valid/test split are missing:", missing)
    print(f"Reading {len(years)} vintages from {RAW_DIR}\n")

    frames, cutoff = [], 0
    for year in years:
        t0 = time.time()
        orig, perf = load_orig(year), load_perf(year)
        loans = add_first_default(orig, perf)[KEEP]
        for col in KEEP_CATEGORICAL:  # one consistent text type across years
            loans[col] = loans[col].astype("string")
        cutoff = max(cutoff, int(perf["period"].max()))
        frames.append(loans)
        print(f"{year}: {len(loans):>6,} loans | {len(perf):>9,} perf rows | default within "
              f"{WINDOW}m {(loans['first_default_mob'] <= WINDOW).mean():6.2%} | "
              f"{time.time() - t0:4.0f}s")
        del perf, orig

    loans = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(LOANS_PATH, index=False)
    META_PATH.write_text(json.dumps({"performance_cutoff": cutoff, "years": years}, indent=2))
    print(f"\nSaved {len(loans):,} loans to {LOANS_PATH}")
    print(f"Performance data runs to {cutoff} (used to drop loans whose window is incomplete)")


if __name__ == "__main__":
    main()
