"""Step 3: out-of-time evaluation: discrimination, calibration, uncertainty and stability.

Run:  python src/evaluate.py
Output: reports/results.md, reports/*.csv, reports/figures/*.png
"""
import matplotlib

matplotlib.use("Agg")  # write figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

from config import (COVID_FROM_YEAR, FIG_DIR, MODEL_LABELS, MODEL_NAMES, PREDICTIONS_PATH,
                    REPORTS_DIR, SEED, TEST_YEARS, TRAIN_YEARS, VALID_YEARS, WINDOW)
from features import NUMERIC, load_modelling_data

N_BOOT = 500
BASELINE = "lr_linear"
EPS = 1e-6
COLORS = {"lr_linear": "#1f77b4", "lr_binned": "#17becf", "lgbm": "#d62728", "lgbm_monotone": "#ff7f0e"}


# ---------- metrics ----------

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ks_stat(y, p):
    """Kolmogorov-Smirnov: largest gap between the score distributions of defaults and non-defaults."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def capture_rate(y, p, top=0.10):
    """Share of all defaults found in the riskiest `top` fraction of loans."""
    riskiest = np.argsort(-p, kind="stable")[: int(np.ceil(top * len(p)))]
    return y[riskiest].sum() / y.sum()


def calibration_slope(y, p):
    """Slope b in logit P(default) = a + b * logit(p). b < 1: predictions too extreme."""
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(fit.coef_[0, 0])


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "loans": len(y), "defaults": int(y.sum()), "default_rate": y.mean(), "mean_pd": p.mean(),
        "auc": auc, "gini": 2 * auc - 1, "pr_auc": average_precision_score(y, p),
        "ks": ks_stat(y, p), "capture_top10": capture_rate(y, p),
        "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
        "obs_to_exp": y.mean() / p.mean(), "cal_slope": calibration_slope(y, p),
    }


def bootstrap_auc_diff(y, p_model, p_base, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap: resample loans, recompute both AUCs on the same resample."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[b] = roc_auc_score(y[idx], p_model[idx]) - roc_auc_score(y[idx], p_base[idx])
    return diffs


def psi(expected, actual, n_bins=10):
    """Population stability index, bins set on the expected (training) distribution."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.nanquantile(e, np.linspace(0, 1, n_bins + 1)[1:-1]))

    def shares(x):
        bins = np.digitize(x, edges)
        bins[np.isnan(x)] = -1  # missing values as their own bin
        return pd.Series(bins).value_counts(normalize=True)

    s = pd.concat([shares(e), shares(a)], axis=1).fillna(0).clip(lower=1e-4)
    return float(((s.iloc[:, 1] - s.iloc[:, 0]) * np.log(s.iloc[:, 1] / s.iloc[:, 0])).sum())


def psi_label(v):
    return "stable" if v < 0.10 else ("moderate shift" if v < 0.25 else "major shift")


# ---------- output helpers ----------

def md_table(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join([str(df.index.name or "")] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + ["" if pd.isna(row[c]) else fmt.get(c, "{}").format(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


FMT = {"loans": "{:,.0f}", "defaults": "{:,.0f}", "default_rate": "{:.2%}", "mean_pd": "{:.2%}",
       "auc": "{:.3f}", "gini": "{:.3f}", "pr_auc": "{:.3f}", "ks": "{:.3f}",
       "capture_top10": "{:.1%}", "brier": "{:.5f}", "log_loss": "{:.4f}",
       "obs_to_exp": "{:.2f}", "cal_slope": "{:.2f}"}


def years_label(years):
    return f"{min(years)}-{max(years)}"


# ---------- figures ----------

def plot_roc_pr(y, preds, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in MODEL_NAMES:
        p = preds[f"p_{m}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=COLORS[m], label=f"{MODEL_LABELS[m]} (AUC {roc_auc_score(y, p):.3f})")
        axes[1].plot(rec, prec, color=COLORS[m], label=f"{MODEL_LABELS[m]} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"Base rate {y.mean():.2%}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(f"Out-of-time test, vintages {years_label(TEST_YEARS)}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability(y, p, n_bins=10):
    bins = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    g = pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin")
    return g["p"].mean().to_numpy(), g["y"].mean().to_numpy()


def plot_calibration(y, raw, recal, path):
    curves = {(k, m): reliability(y, src[m]) for k, src in [("raw", raw), ("recal", recal)] for m in MODEL_NAMES}
    values = np.concatenate([np.r_[a, b] for a, b in curves.values()])
    values = values[values > 0]
    lims = [max(values.min() / 2, 1e-5), min(values.max() * 2, 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, key, title in [(axes[0], "raw", "As trained"),
                           (axes[1], "recal", f"Recalibrated on {years_label(VALID_YEARS)}")]:
        for m in MODEL_NAMES:
            pred, obs = curves[(key, m)]
            obs = np.where(obs > 0, obs, np.nan)  # a bin with no defaults cannot go on a log axis
            ax.plot(pred, obs, "o-", color=COLORS[m], ms=4, label=MODEL_LABELS[m])
        ax.plot(lims, lims, "k--", lw=0.8, label="Perfect calibration")
        ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims, title=title,
               xlabel="Mean predicted PD (decile of predictions)", ylabel="Observed default rate")
        ax.legend(fontsize=8)
    fig.suptitle(f"Calibration on the out-of-time test ({years_label(TEST_YEARS)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vintages(rates, aucs, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    ax.plot(rates.index, rates["default_rate"], "k-o", lw=2, ms=4, label="Observed")
    for m in MODEL_NAMES:
        ax.plot(rates.index, rates[m], "--", color=COLORS[m], label=f"Predicted: {MODEL_LABELS[m]}")
    ax.set(ylabel=f"Default rate within {WINDOW} months", title="Observed vs predicted default rate by vintage")
    ax.legend(fontsize=8)
    ax = axes[1]
    for m in MODEL_NAMES:
        ax.plot(aucs.index, aucs[m], "-o", color=COLORS[m], ms=4, label=MODEL_LABELS[m])
    ax.set(ylabel="AUC", xlabel="Origination year", title="Discrimination by vintage (after training years)")
    ax.legend(fontsize=8)
    has_covid = rates.index.max() >= COVID_FROM_YEAR
    for a in axes:
        a.axvspan(min(TRAIN_YEARS) - 0.5, max(TRAIN_YEARS) + 0.5, color="grey", alpha=0.08)
        a.axvspan(min(VALID_YEARS) - 0.5, max(VALID_YEARS) + 0.5, color="green", alpha=0.08)
        a.axvspan(min(TEST_YEARS) - 0.5, max(TEST_YEARS) + 0.5, color="blue", alpha=0.08)
        if has_covid:
            a.axvspan(COVID_FROM_YEAR - 0.5, rates.index.max() + 0.5, color="red", alpha=0.05)
    top = axes[0].get_ylim()[1]
    labels = [(np.mean(TRAIN_YEARS), "train"), (np.mean(VALID_YEARS), "valid"), (np.mean(TEST_YEARS), "test")]
    if has_covid:
        labels.append(((COVID_FROM_YEAR + rates.index.max()) / 2, "COVID-affected"))
    for x, text in labels:
        axes[0].text(x, top * 0.97, text, ha="center", va="top", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- main ----------

def main():
    preds = pd.read_parquet(PREDICTIONS_PATH)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    by_split = {s: preds[preds["split"] == s] for s in ["train", "test_in_time", "valid", "test", "monitor"]}
    test, valid = by_split["test"], by_split["valid"]
    y_te = test["default"].to_numpy()
    out = []

    # 1. Data summary
    summary = pd.DataFrame({
        "vintages": {"train": years_label(TRAIN_YEARS), "test_in_time": years_label(TRAIN_YEARS),
                     "valid": years_label(VALID_YEARS), "test": years_label(TEST_YEARS),
                     "monitor": f"{max(TEST_YEARS) + 1}+" if len(by_split["monitor"]) else "-"},
        "loans": {s: len(d) for s, d in by_split.items()},
        "defaults": {s: int(d["default"].sum()) for s, d in by_split.items()},
        "default_rate": {s: d["default"].mean() for s, d in by_split.items()},
    })
    summary.index.name = "split"
    out += ["## Data", md_table(summary, FMT)]

    # 2. Out-of-time test performance (as trained)
    oot = pd.DataFrame({MODEL_LABELS[m]: metrics(y_te, test[f"p_{m}"].to_numpy()) for m in MODEL_NAMES}).T
    oot.index.name = "model"
    oot.to_csv(REPORTS_DIR / "metrics_oot.csv")
    disc = oot[["auc", "gini", "pr_auc", "ks", "capture_top10"]]
    out += [f"## Out-of-time test ({years_label(TEST_YEARS)}): discrimination", md_table(disc, FMT)]

    # 3. In-time vs out-of-time
    it = by_split["test_in_time"]
    rows = {}
    for m in MODEL_NAMES:
        a_in = roc_auc_score(it["default"], it[f"p_{m}"])
        a_out = roc_auc_score(y_te, test[f"p_{m}"])
        rows[MODEL_LABELS[m]] = {"auc_in_time": a_in, "auc_out_of_time": a_out, "change": a_out - a_in}
    inout = pd.DataFrame(rows).T
    inout.index.name = "model"
    inout.to_csv(REPORTS_DIR / "in_time_vs_oot.csv")
    out += ["## In-time vs out-of-time AUC",
            "In-time = random 20% of loans held out from the training vintages.",
            md_table(inout, {c: "{:+.3f}" if c == "change" else "{:.3f}" for c in inout.columns})]

    # 4. Paired bootstrap of the AUC gain over the baseline
    print(f"Bootstrapping AUC differences ({N_BOOT} resamples per model)...")
    rows = {}
    for m in MODEL_NAMES:
        if m == BASELINE:
            continue
        d = bootstrap_auc_diff(y_te, test[f"p_{m}"].to_numpy(), test[f"p_{BASELINE}"].to_numpy())
        point = roc_auc_score(y_te, test[f"p_{m}"]) - roc_auc_score(y_te, test[f"p_{BASELINE}"])
        rows[MODEL_LABELS[m]] = {"auc_gain": point, "ci_low": np.percentile(d, 2.5),
                                 "ci_high": np.percentile(d, 97.5), "share_resamples_gain_gt_0": (d > 0).mean()}
    boot = pd.DataFrame(rows).T
    boot.index.name = "model"
    boot.to_csv(REPORTS_DIR / "bootstrap_auc.csv")
    out += [f"## AUC gain over {MODEL_LABELS[BASELINE].lower()} (paired bootstrap, 95% CI)",
            md_table(boot, {"auc_gain": "{:+.4f}", "ci_low": "{:+.4f}", "ci_high": "{:+.4f}",
                            "share_resamples_gain_gt_0": "{:.1%}"})]

    # 5. Calibration, before and after recalibration on the validation vintages
    raw, recal, rows = {}, {}, {}
    for m in MODEL_NAMES:
        # Platt scaling: a 2-parameter logistic map from old log-odds to new probabilities.
        # It is monotone, so the ranking (and AUC) is unchanged; only the level and spread move.
        rc = LogisticRegression(C=1e6, max_iter=1000).fit(logit(valid[f"p_{m}"].to_numpy()).reshape(-1, 1),
                                                          valid["default"])
        raw[m] = test[f"p_{m}"].to_numpy()
        recal[m] = rc.predict_proba(logit(raw[m]).reshape(-1, 1))[:, 1]
        for label, p in [("as trained", raw[m]), ("recalibrated", recal[m])]:
            r = metrics(y_te, p)
            rows[(MODEL_LABELS[m], label)] = {k: r[k] for k in ["mean_pd", "default_rate", "obs_to_exp",
                                                               "cal_slope", "brier", "log_loss"]}
    cal = pd.DataFrame(rows).T
    cal.index = [f"{a} - {b}" for a, b in cal.index]
    cal.index.name = "model"
    cal.to_csv(REPORTS_DIR / "calibration.csv")
    out += [f"## Calibration on the out-of-time test (recalibration fitted on {years_label(VALID_YEARS)})",
            "obs_to_exp = observed / mean predicted default rate (1 is perfect); cal_slope 1 is perfect.",
            md_table(cal, FMT)]

    # 6. Stability by vintage
    rates = preds.groupby("vintage").agg(default_rate=("default", "mean"),
                                         **{m: (f"p_{m}", "mean") for m in MODEL_NAMES})
    later = preds[preds["vintage"] >= min(VALID_YEARS)]
    auc_rows = {}
    for v, g in later.groupby("vintage"):
        if g["default"].nunique() < 2:
            continue
        auc_rows[v] = {m: roc_auc_score(g["default"], g[f"p_{m}"]) for m in MODEL_NAMES}
        auc_rows[v].update({"loans": len(g), "defaults": int(g["default"].sum()), "default_rate": g["default"].mean()})
    aucs = pd.DataFrame(auc_rows).T
    aucs.index.name = "vintage"
    aucs.to_csv(REPORTS_DIR / "metrics_by_vintage.csv")
    vint = aucs[["loans", "defaults", "default_rate"] + MODEL_NAMES].rename(columns=MODEL_LABELS)
    vint.index = [f"{v}*" if v >= COVID_FROM_YEAR else str(v) for v in vint.index]
    vint.index.name = "vintage"
    out += ["## AUC by vintage",
            f"\\* {WINDOW}-month window overlaps the COVID-19 forbearance period; read with caution.",
            md_table(vint, {**FMT, **{MODEL_LABELS[m]: "{:.3f}" for m in MODEL_NAMES}})]

    # 7. Population stability: training vintages vs out-of-time test
    df = load_modelling_data()
    tr_feat, te_feat = df[df["split"] == "train"], df[df["split"] == "test"]
    rows = {f: {"psi": psi(tr_feat[f], te_feat[f])} for f in NUMERIC}
    rows.update({f"score: {MODEL_LABELS[m]}": {"psi": psi(by_split["train"][f"p_{m}"], test[f"p_{m}"])}
                 for m in MODEL_NAMES})
    stab = pd.DataFrame(rows).T
    stab["reading"] = stab["psi"].map(psi_label)
    stab.index.name = "variable"
    stab.to_csv(REPORTS_DIR / "psi.csv")
    out += [f"## Population stability index: training vintages vs {years_label(TEST_YEARS)}",
            "Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.",
            md_table(stab, {"psi": "{:.3f}"})]

    # Figures
    plot_roc_pr(y_te, test, FIG_DIR / "roc_pr_oot.png")
    plot_calibration(y_te, raw, recal, FIG_DIR / "calibration_oot.png")
    plot_vintages(rates, aucs[MODEL_NAMES], FIG_DIR / "vintage_stability.png")

    header = (f"# Results\n\nGenerated by `src/evaluate.py`. Default = 90+ days delinquent, REO, "
              f"or a credit-related termination within {WINDOW} months of first payment.\n")
    (REPORTS_DIR / "results.md").write_text(header + "\n\n".join(out) + "\n", encoding="utf-8")

    pd.set_option("display.width", 160)
    print("\nOut-of-time test:\n", oot[["auc", "pr_auc", "ks", "capture_top10", "brier", "obs_to_exp", "cal_slope"]].round(4).to_string())
    print("\nIn-time vs out-of-time AUC:\n", inout.round(4).to_string())
    print("\nAUC gain over baseline:\n", boot.round(4).to_string())
    print("\nCalibration:\n", cal[["mean_pd", "default_rate", "obs_to_exp", "cal_slope", "brier"]].round(4).to_string())
    print("\nAUC by vintage:\n", aucs.round(3).to_string())
    print("\nPSI:\n", stab.round(3).to_string())
    print(f"\nWrote {REPORTS_DIR / 'results.md'} and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
