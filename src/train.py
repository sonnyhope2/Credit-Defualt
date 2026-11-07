"""Step 2: fit four models on the training vintages and choose settings on the validation vintages.

Run:  python src/train.py
Output: models/*.joblib, data/processed/predictions.parquet, reports/model_selection.csv
"""
import inspect
import time

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import MODEL_NAMES, MODELS_DIR, PREDICTIONS_PATH, REPORTS_DIR
from features import FEATURES, load_modelling_data
from modelling import make_lgbm, make_lr_binned, make_lr_linear

# LightGBM 4.7 renamed the validation-set argument; support both old and new versions
_NEW_EVAL_API = "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters

LR_C_GRID = [0.01, 0.1, 1.0]  # inverse regularisation strength
LGBM_GRID = [dict(num_leaves=nl, min_child_samples=mc) for nl in (15, 63) for mc in (200, 1000)]
EARLY_STOPPING_ROUNDS = 200


def fit_lgbm(pipe, X_tr, y_tr, X_va, y_va):
    """Stop adding trees once validation AUC has not improved for EARLY_STOPPING_ROUNDS rounds."""
    X_va_cast = pipe.named_steps["cast"].fit(X_tr).transform(X_va)
    if _NEW_EVAL_API:
        eval_args = {"model__eval_X": (X_va_cast,), "model__eval_y": (y_va,)}
    else:
        eval_args = {"model__eval_set": [(X_va_cast, y_va)]}
    pipe.fit(X_tr, y_tr, **eval_args,
             model__callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return pipe


def main():
    df = load_modelling_data()
    tr, va = df[df["split"] == "train"], df[df["split"] == "valid"]
    X_tr, y_tr, X_va, y_va = tr[FEATURES], tr["default"], va[FEATURES], va["default"]
    print(f"train: {len(tr):,} loans ({y_tr.mean():.2%} default) | "
          f"valid: {len(va):,} loans ({y_va.mean():.2%} default)\n")

    rows, best = [], {}

    def record(name, settings, pipe, t0, params=None):
        auc = roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1])
        rows.append({"model": name, "settings": settings, "valid_auc": round(auc, 4),
                     "seconds": round(time.time() - t0)})
        print(f"  {name:<14} {settings:<45} valid AUC {auc:.4f} ({time.time() - t0:.0f}s)")
        if name not in best or auc > best[name]["auc"]:
            best[name] = {"auc": auc, "pipe": pipe, "params": params}

    for name, make in [("lr_linear", make_lr_linear), ("lr_binned", make_lr_binned)]:
        for C in LR_C_GRID:
            t0 = time.time()
            record(name, f"C={C}", make(C=C).fit(X_tr, y_tr), t0)

    for params in LGBM_GRID:
        t0 = time.time()
        pipe = fit_lgbm(make_lgbm(**params), X_tr, y_tr, X_va, y_va)
        record("lgbm", f"leaves={params['num_leaves']}, min_child={params['min_child_samples']}, "
               f"trees={pipe[-1].best_iteration_}", pipe, t0, params)

    # Same settings as the best LightGBM plus monotone constraints, so any difference
    # between the two comes from the constraints alone
    t0 = time.time()
    params = best["lgbm"]["params"]
    pipe = fit_lgbm(make_lgbm(**params, monotone=True), X_tr, y_tr, X_va, y_va)
    record("lgbm_monotone", f"leaves={params['num_leaves']}, min_child={params['min_child_samples']}, "
           f"trees={pipe[-1].best_iteration_}", pipe, t0, params)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds = df[["loan_id", "vintage", "split", "default"]].copy()
    print("\nChosen settings:")
    for name in MODEL_NAMES:
        joblib.dump(best[name]["pipe"], MODELS_DIR / f"{name}.joblib")
        preds[f"p_{name}"] = best[name]["pipe"].predict_proba(df[FEATURES])[:, 1]
        chosen = [r for r in rows if r["model"] == name and r["valid_auc"] == round(best[name]["auc"], 4)][0]
        print(f"  {name:<14} {chosen['settings']}")
    preds.to_parquet(PREDICTIONS_PATH, index=False)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "model_selection.csv", index=False)
    print(f"\nSaved models to {MODELS_DIR} and predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
