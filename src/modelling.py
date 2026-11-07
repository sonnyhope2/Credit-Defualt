"""The four models, each a scikit-learn Pipeline so preprocessing is learned on training data only."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import SEED
from features import BINARY, CATEGORICAL, FEATURES, MONOTONE, NUMERIC


class QuantileBinner(BaseEstimator, TransformerMixin):
    """Cut each numeric column at quantiles learned on the training data (scorecard-style bins).

    Missing values get their own bin (-1), as in a credit scorecard, so no imputation is needed.
    """

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(
            getattr(X, "columns", [f"x{i}" for i in range(np.shape(X)[1])]), dtype=object)
        X = np.asarray(X, dtype=float)
        inner = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.nanquantile(col, inner)) for col in X.T]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape, dtype=int)
        for j, edges in enumerate(self.edges_):
            bins = np.digitize(X[:, j], edges)
            bins[np.isnan(X[:, j])] = -1
            out[:, j] = bins
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Turn text columns into pandas categoricals so LightGBM splits on them natively.

    Levels are fixed at fit time; a level first seen later (e.g. a new state code) becomes missing.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.levels_ = {c: sorted(pd.unique(X[c])) for c in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X[c] = pd.Categorical(X[c], categories=self.levels_[c])
        return X


def _lr_pipeline(numeric_step, C):
    pre = ColumnTransformer([
        ("num", numeric_step, NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=C, max_iter=3000))])


def make_lr_linear(C=1.0):
    """Classic baseline: one coefficient per numeric feature (median-imputed, standardised,
    plus a missing flag) and one per category level."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    return _lr_pipeline(numeric, C)


def make_lr_binned(C=1.0, n_bins=10):
    """Scorecard-style logistic regression: numeric features cut into deciles, so the model
    can bend (e.g. risk flat above FICO 760, steep below 680) while staying additive."""
    numeric = Pipeline([("bins", QuantileBinner(n_bins)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return _lr_pipeline(numeric, C)


LGBM_BASE = dict(n_estimators=5000, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, metric="auc", random_state=SEED,
                 n_jobs=-1, verbose=-1)


def make_lgbm(num_leaves=31, min_child_samples=200, monotone=False):
    """Gradient boosting. NaNs and categories are handled natively. No class reweighting:
    reweighting would distort the predicted probabilities we later check for calibration."""
    params = dict(LGBM_BASE, num_leaves=num_leaves, min_child_samples=min_child_samples)
    if monotone:
        params["monotone_constraints"] = [MONOTONE.get(f, 0) for f in FEATURES]
    return Pipeline([("cast", CategoryCaster(CATEGORICAL)), ("model", LGBMClassifier(**params))])
