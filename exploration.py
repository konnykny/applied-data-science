
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Exploration(BaseEstimator, TransformerMixin):
    """
    Light EDA within a pipeline. Returns the input DataFrame unchanged.
    Parameters
    ----------
    nan_report : bool
        If True, prints % missing per column (only those with missing).
    plot_data : str
        'All' | 'weather' | 'energy' | 'seaborn' | None. If set, produces time-series plots.
        'seaborne' is accepted and treated as 'seaborn'.
    """
    nan_report: bool = True
    plot_data: Optional[str] = "All"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Exploration expects a pandas DataFrame.")

        # print basic info (columns, first few rows)
        print("\n[Exploration] DataFrame info:")
        print(X.info())
        print("\n[Exploration] First few rows:")
        print(X.head().to_string())


        if self.nan_report:
            na_pct = X.isna().mean().loc[lambda s: s.gt(0)].sort_values(ascending=False) * 100.0
            if not na_pct.empty:
                print("\n[Exploration] Missing value report (% of rows with NaN):")
                print(na_pct.round(2).to_string())
            else:
                print("\n[Exploration] No missing values detected.")

        # all-zero columns report
        numeric = X.select_dtypes(include=[np.number])
        if numeric.empty:
            print("\n[Exploration] All-zero report: no numeric columns to check.")
        else:
            all_zero_cols = []
            for col in numeric.columns:
                s = pd.to_numeric(numeric[col], errors="coerce")
                # consider a column "all zero" when all non-NA values equal 0
                if not s.isna().all() and (s.fillna(0) == 0).all():
                    all_zero_cols.append(col)
            if all_zero_cols:
                print("\n[Exploration] All-zero columns (all non-NA entries == 0):")
                for c in all_zero_cols:
                    print(f" - {c}")
            else:
                print("\n[Exploration] No all-zero columns found.")




        if self.plot_data is not None:
            self._plot(X)

        ### debugging plot
        # if "generation nuclear" in X.columns:
        #     plt.figure(figsize=(10, 4))
        #     ax = X["generation nuclear"].plot()
        #     ax.set_title("generation_nuclear over time")
        #     ax.set_xlabel("Time")
        #     ax.set_ylabel("Value")
        #     plt.tight_layout()
        #     plt.show()
        # else:
        #     print("[Exploration] 'generation nuclear' column not found; skipping debug plot.")

        return X

    def _plot(self, df: pd.DataFrame):
        energy_cols = [c for c in df.columns if any(k in c.lower() for k in [
            "load", "consumption", "demand", "generation", "solar", "wind", "hydro", "coal",
            "gas", "nuclear", "biomass", "geothermal", "renew", "non_renew", "usage", "power",
            "total_renewable_generation", "total_fossil_generation"
        ])]
        weather_cols = [c for c in df.columns if c not in energy_cols]

        seaborn_mode = str(self.plot_data).lower() in ("seaborn", "seaborne")

        if seaborn_mode:
            try:
                import seaborn as sns
            except Exception:
                print("[Exploration] seaborn not available; install `seaborn` for seaborn plots.")
                seaborn_mode = False

        if seaborn_mode: ## dont use it its garbage
            sns.set_style("whitegrid")

            def _plot_group_seaborn(cols, title):
                if not cols:
                    return
                # prepare long form with a time column
                df_group = df[cols].copy()
                df_reset = df_group.reset_index()
                time_col = df_reset.columns[0]
                # try to ensure datetime for nicer x-axis
                try:
                    df_reset[time_col] = pd.to_datetime(df_reset[time_col])
                except Exception:
                    pass
                melted = df_reset.melt(id_vars=[time_col], var_name="variable", value_name="value")

                plt.figure(figsize=(16, 6))
                ax = sns.lineplot(data=melted, x=time_col, y="value", hue="variable", lw=1)
                ax.set_title(f"{title} columns over time (seaborn)")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                plt.tight_layout()

                # interactive hover if mplcursors is available
                try:
                    import mplcursors
                    cursor = mplcursors.cursor(ax.lines, hover=True)

                    @cursor.connect("add")
                    def on_add(sel):
                        # sel.target is (x, y)
                        x, y = sel.target
                        label = sel.artist.get_label()
                        sel.annotation.set_text(f"{label}\n{x}: {y:.3f}")
                except Exception:
                    print("[Exploration] mplcursors not available — hover tooltips disabled for seaborn plots.")

                plt.show()

            _plot_group_seaborn(energy_cols, "Energy-related")
            _plot_group_seaborn(weather_cols, "Weather / other")
            return

        # fallback / original matplotlib plotting behavior
        if self.plot_data in ("All", "energy"):
            if energy_cols:
                plt.figure(figsize=(16, 18))
                df[energy_cols].plot(ax=plt.gca())
                plt.title("Energy-related columns over time")
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.tight_layout()
                plt.show()

        if self.plot_data in ("All", "weather") and weather_cols:
            n = len(weather_cols)
            fig, axes = plt.subplots(n, 1, figsize=(16, max(2 * n, 6)), sharex=True)
            if n == 1:
                axes = [axes]
            for ax, col in zip(axes, weather_cols):
                df[col].plot(ax=ax)
                ax.set_title(col)
                ax.set_xlabel("")
            axes[-1].set_xlabel("Time")
            plt.tight_layout()
            plt.show()