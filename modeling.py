from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb

X_forbidden_cols = [
    'total load actual',
    'price actual',
    'price day ahead',
    'total load forecast',
    # 'generation wind onshore',
    # 'generation waste',
    #
    # 'generation biomass',
    # 'generation fossil brown coal/lignite',
    # 'generation fossil gas',
    # 'generation fossil hard coal',
    # 'generation fossil oil',
    # 'generation hydro pumped storage consumption',
    # 'generation hydro run-of-river and poundage',
    # 'generation hydro water reservoir',
    # 'generation nuclear',
    # 'generation other',
    # 'generation other renewable',
    # 'generation solar',

    'total_fossil_generation',
    'total_renewable_generation',
    # 'renewable_generation_ratio',

    # 'time__original_tz',
]


def _chronological_split(X: pd.DataFrame, y: pd.Series, time: pd.Series, test_size: float = 0.2):
    n = len(X)
    n_test = max(1, int(n * test_size))
    split = n - n_test
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], time.iloc[:split], time.iloc[split:]


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
    target_column: Optional[str] = 'renewable_generation_ratio'
    test_size: float = 0.2
    random_state: int = 42
    prediction_horizon: int = 24
    plot_results: bool = True
    prediction_hour: int = 11

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            if self.target_column is None or self.target_column not in X.columns:
                raise ValueError("Provide y or specify target_column present in X.")
            y = X[self.target_column]
            X = X.drop(columns=[self.target_column])
            X = X.drop(columns=X_forbidden_cols, errors='ignore')

        data = pd.concat([X, y], axis=1).dropna(axis=0, how="any")
        # remove the datetime columns
        #for col in data.columns:
        #    print(f'{col} -> {data[col].dtype}')

        time_col = pd.to_datetime(data['time__original_tz'], utc=True).copy(deep=True)
        print(time_col.dtype)
        print(time_col)
        data = data.drop(columns=['time__original_tz'])

        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]

        X, y, time_col = self._preprocess_multi_target(X, y, time_col)
        print(X.shape, y.shape)
        print(time_col.dtype)

        X_tr, X_te, y_tr, y_te, time_tr, time_te = _chronological_split(X, y, time_col, test_size=self.test_size)
        print(time_te.dtype)

        base_models = {
            "XGBoost": xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=self.random_state, n_jobs=-1, verbose=2),
            #"LinearRegression": LinearRegression(),
            #"RandomForest": RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1),
        }

        print(f'target col: {self.target_column}')
        print(f'traning cols ({len(X.columns)})')
        # for col in X.columns:
        #     print(f'   {col} ({X[col].dtype})-> {X[col].isna().sum()}')
        print(X.info())

        scores = {}
        fitted = {}
        predictions = {}
        pred = None
        for name, base_model in base_models.items():

            # wrap 1 model -> one horizon
            model = MultiOutputRegressor(base_model, n_jobs=-1)
            model.fit(X_tr, y_tr)

            # predict
            pred = model.predict(X_te)

            # get the overall metrics
            r2 = r2_score(y_te, pred)
            mae = mean_absolute_error(y_te, pred)
            rmse = root_mean_squared_error(y_te, pred)

            # get the metrics per individual predictoon horizon
            horizon_scores = {}
            for i in range(self.prediction_horizon):
                h_r2 = r2_score(y_te.iloc[:, i], pred[:, i])
                h_mae = mean_absolute_error(y_te.iloc[:, i], pred[:, i])
                h_rmse = root_mean_squared_error(y_te.iloc[:, i], pred[:, i])
                horizon_scores[i] = {"R2": h_r2, "MAE": h_mae, "RMSE": h_rmse}

            # store
            fitted[name] = model
            scores[name] = {
                "Overall": {"R2": r2, "MAE": mae, "RMSE": rmse},
                "ByHorizon": horizon_scores
            }
            predictions[name] = pred
            print(f"[ModelTraining] {name} (overall): R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
            for i in range(self.prediction_horizon):
                print(f'    [horizon t+{i+1}]: R2={horizon_scores[i]["R2"]:.4f}  MAE={horizon_scores[i]["MAE"]:.4f}  RMSE={horizon_scores[i]["RMSE"]:.4f}')

        best_name = min(scores, key=lambda k: scores[k]["Overall"]["RMSE"])
        self.best_estimator_ = fitted[best_name]
        self.best_scores_ = scores
        self.feature_names_in_ = X.columns

        #if self.plot_results:
        #    pred = self.best_estimator_.predict(X_te)
        #    _plot_metrics(y_te, pd.Series(pred, index=y_te.index), title=f"Best: {best_name}")

        print('saving predictions')
        with open('./test_results.pkl', 'wb') as f:
            pickle.dump({
                'x': X_te,
                'preds': pred,
                'gt': y_te,
                'adjusted_preds': self._postprocess_predictions(pred, time_te),
                'adjusted_gt': self._postprocess_predictions(y_te, time_te),
                'time': time_col,
                'pred_time': self._postprocess_predictions(time_te.copy(deep=True), time_te)
            }, f)

        print('saving the models')
        with open('./models.pkl', 'wb') as f:
            pickle.dump(fitted, f)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray: # transform = predict
        if not hasattr(self, "best_estimator_"):
            raise RuntimeError("ModelTraining not fitted yet.")
        X = X.reindex(columns=self.feature_names_in_, fill_value=0)
        return self.best_estimator_.predict(X)


    def _preprocess_multi_target(self, X: pd.DataFrame, y_single: pd.DataFrame, t: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

        y_multi = pd.DataFrame()
        print(self.prediction_horizon)
        for i in range(1, self.prediction_horizon + 1):
            y_multi[f'y_t+{i}'] = y_single.shift(-i)

        y_multi = y_multi.iloc[:-self.prediction_horizon]
        X_multi = X.iloc[:-self.prediction_horizon]
        t_multi = pd.to_datetime(t.iloc[:-self.prediction_horizon])

        return X_multi, y_multi, t_multi

    def _postprocess_predictions(self, pred: pd.DataFrame, time: pd.Series) -> pd.DataFrame:
        time_mask = time.dt.hour == self.prediction_hour
        return pred[time_mask]


class  MultiRegXGBoostTraining(ModelTraining):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Train a multi-output XGBoost model to predict the 24-hour window targets.

        Behavior:
        - If X already contains columns named like 'future_{target_column}_<num>' these are used
          (sorted by trailing integer) as the multi-target y. Those columns are removed from X
          before training.
        - Otherwise, fall back to the previous behavior: take `self.target_column` from X and
          construct multi-target horizon columns by shifting.
        - Trains a MultiOutputRegressor wrapped XGBoost model and computes a feature importance
          report (mean importance across targets) saved to './xgb_feature_importance.csv'.
        """

        X = X.copy()

        # get target columns
        future_prefix = f'future_{self.target_column}_f'
        future_cols = [c for c in X.columns if c.startswith(future_prefix)]
        n_targets = len(future_cols)

        if 'time__original_tz' in X.columns:
            # time column may be present in X; keep for splitting
            time_col = pd.to_datetime(X['time__original_tz'], utc=True).copy(deep=True)
        else:
            raise ValueError("time__original_tz column required in data for chronological split")

        y = pd.DataFrame(X[future_cols].copy())
        X = X.drop(columns=future_cols)
        X = X.drop(columns=['time__original_tz'])
        X = X.drop(columns=X_forbidden_cols)

        # drop rows with any NaN (user previously used dropna)
        X = X.dropna(axis=0, how='any')
        X = X.drop(columns=['Unnamed: 0'], errors='ignore') # exists when loaded from pickle sometimes

        print("data_____________")
        print(X)

        print("y_____________")
        print(y)

        # X_final = X
        # y_final = y

        # chronological split
        X_tr, X_te, y_tr, y_te, time_tr, time_te = _chronological_split(X, y, time_col.loc[X.index], test_size=self.test_size)

        # Train multioutput XGBoost via sklearn wrapper
        base = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=self.random_state, verbosity=1, n_jobs=-1,
            objective='reg:squarederror'
        )

        print("[MultiRegXGBoostTraining] Training MultiOutputRegressor with XGBoost base...")

        model = MultiOutputRegressor(base, n_jobs=-1)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        self.pred = y_pred

        # overall metrics
        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = root_mean_squared_error(y_te, y_pred)

        print(f"[MultiRegXGBoostTraining] Overall: R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")



        # --- New: compute and visualise R2 per prediction horizon ---
        try:
            horizon_r2 = []
            for i in range(n_targets):
                h_r2 = r2_score(y_te.iloc[:, i], y_pred[:, i])
                horizon_r2.append(h_r2)

            plt.figure(figsize=(10, 4))
            x = list(range(1, n_targets + 1))
            plt.bar(x, horizon_r2, color='C0', alpha=0.8)
            plt.xlabel('Horizon (hours ahead)')
            plt.ylabel('R2')
            plt.title('[MultiRegXGBoostTraining] R2 per predicted hour')
            plt.xticks(x)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('./r2_by_horizon.png')
            plt.close()
            print('[MultiRegXGBoostTraining] Saved R2 per horizon plot to ./r2_by_horizon.png')
        except Exception as e:
            print('[MultiRegXGBoostTraining] Could not create R2 per-horizon plot:', e)

        # feature importance report: average importance across outputs
        try:
            importances = np.vstack([est.feature_importances_ for est in model.estimators_])
            mean_imp = np.mean(importances, axis=0)
            sum_imp = np.sum(importances, axis=0)
            imp_df = pd.DataFrame(importances, index=[f'target_{i}' for i in range(importances.shape[0])], columns=X_tr.columns)
            # add summary rows
            imp_df.loc['mean'] = mean_imp
            imp_df.loc['sum'] = sum_imp

            # transpose so features are rows; then put 'sum' column first, then target-specific columns, then 'mean'
            out_df = imp_df.T
            target_cols = [f'target_{i}' for i in range(importances.shape[0])]
            cols_order = ['sum'] + target_cols + ['mean']
            # some safety: keep only existing columns (in case of naming mismatch)
            cols_order = [c for c in cols_order if c in out_df.columns]
            out_df = out_df[cols_order]

            out_df.sort_values(by='sum', ascending=False).to_csv('./xgb_feature_importance.csv')
            print('[MultiRegXGBoostTraining] Saved feature importance report to ./xgb_feature_importance.csv')
        except Exception as e:
            print('[MultiRegXGBoostTraining] Could not compute feature importances:', e)

        # store results
        self.best_estimator_ = model
        self.best_scores_ = {
            "Overall": {"R2": r2, "MAE": mae, "RMSE": rmse}
        }
        self.feature_names_in_ = X.columns

        # save predictions and model like previous implementation
        with open('./test_results.pkl', 'wb') as f:
            pickle.dump({
                'x': X_te,
                'preds': y_pred,
                'gt': y_te,
                'adjusted_preds': self._postprocess_predictions(pd.DataFrame(y_pred, index=y_te.index), time_te),
                'adjusted_gt': self._postprocess_predictions(y_te, time_te),
                'time': time_col,
                'pred_time': self._postprocess_predictions(time_te.copy(deep=True), time_te)
            }, f)

        with open('./models.pkl', 'wb') as f:
            pickle.dump({'model': model}, f)

        # Build and return a DataFrame indexed by the original time values (time_te)
        try:
            gt_df = pd.DataFrame(y_te).copy()
            pred_df = pd.DataFrame(y_pred, index=y_te.index, columns=[f'pred_{c}' for c in y_te.columns])
            results_df = pd.concat([gt_df, pred_df], axis=1)
            # set index to the datetime index from the chronological split (use original time_col aligned to y_te)
            results_df.index = pd.to_datetime(time_col.loc[y_te.index])
        except Exception:
            return self

        self.results_df = results_df

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray: # transform = predict
        return self.results_df