
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

def _chronological_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    n = len(X)
    n_test = max(1, int(n * test_size))
    split = n - n_test
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _plot_metrics(y_true, y_pred, title="Model Performance"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.title(f"{title}: True vs Pred")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    residuals = y_pred - y_true
    plt.plot(residuals.index, residuals.values, linewidth=1)
    plt.title(f"{title}: Residuals over Time")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(residuals.values, bins=30)
    plt.title(f"{title}: Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


@dataclass
class ModelTraining(BaseEstimator, RegressorMixin):
    target_column: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42
    plot_results: bool = True

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            if self.target_column is None or self.target_column not in X.columns:
                raise ValueError("Provide y or specify target_column present in X.")
            y = X[self.target_column]
            X = X.drop(columns=[self.target_column])

        data = pd.concat([X, y], axis=1).dropna(axis=0, how="any")
        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]

        X_tr, X_te, y_tr, y_te = _chronological_split(X, y, test_size=self.test_size)

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1),
        }
        if _HAS_XGB:
            models["XGBoost"] = XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=self.random_state, n_jobs=-1,
            )

        scores = {}
        fitted = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            r2 = r2_score(y_te, pred)
            mae = mean_absolute_error(y_te, pred)
            rmse = mean_squared_error(y_te, pred, squared=False)
            scores[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
            fitted[name] = model
            print(f"[ModelTraining] {name}: R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

        best_name = min(scores, key=lambda k: scores[k]["RMSE"]) 
        self.best_estimator_ = fitted[best_name]
        self.best_scores_ = scores
        self.feature_names_in_ = X.columns

        if self.plot_results:
            pred = self.best_estimator_.predict(X_te)
            _plot_metrics(y_te, pd.Series(pred, index=y_te.index), title=f"Best: {best_name}")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray: # transform = predict
        if not hasattr(self, "best_estimator_"):
            raise RuntimeError("ModelTraining not fitted yet.")
        X = X.reindex(columns=self.feature_names_in_, fill_value=0)
        return self.best_estimator_.predict(X)
