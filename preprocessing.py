from __future__ import annotations
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Union, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _find_datetime_column(df: pd.DataFrame) -> str:
    """Find a datetime-like column. Prefers columns named 'time' or 'timestamp'."""
    candidates = [c for c in df.columns if c.lower() in {"time", "timestamp", "date", "datetime"}]
    for c in candidates:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            pass
    # fallback: try any column that parses without too many NaT
    for c in df.columns:
        try:
            ser = pd.to_datetime(df[c], errors="coerce")
            if ser.notna().mean() > 0.9:
                return c
        except Exception:
            continue
    raise ValueError("No datetime-like column found. Please ensure a 'time'/'timestamp' column exists.")


def _load_csv(path: str) -> pd.DataFrame:
    """
    Simple UTC-based CSV loader:
    - Detects datetime column via _find_datetime_column
    - Preserves original timestamp string in a new column: <dt_col>__original_tz
    - Parses datetimes to UTC (tz-aware) so all timestamps are used as UTC internally
    - Falls back to treating large integers as epoch seconds (converted to UTC)
    - Returns dataframe with the datetime column set as the DatetimeIndex (tz='UTC')
    """
    df = pd.read_csv(path)
    dt_col = _find_datetime_column(df)

    # preserve original textual timestamp (keeps the offset like +01:00 / +02:00)
    df[f"{dt_col}__original_tz"] = df[dt_col].astype(str).copy(deep=True)

    # Parse timestamps to UTC (handles strings with offsets like +01:00 / +02:00)
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)

    # Allow integer epoch seconds/millis just in case — produce UTC timestamps
    if df[dt_col].isna().all():
        try:
            s = pd.to_numeric(df[dt_col], errors="coerce")
            if s.notna().any():
                # Heuristic: if large, likely ms
                if s.max() > 1e12:
                    s = (s // 1000).astype("Int64")
                df[dt_col] = pd.to_datetime(s, unit="s", errors="coerce", utc=True)
        except Exception:
            pass

    # set index (will be tz-aware UTC if parsing succeeded)
    df = df.set_index(pd.DatetimeIndex(df[dt_col])).sort_index()
    return df

def _flatten_columns(cols: pd.MultiIndex) -> List[str]:
    return [f"{a}__loc_{b}" for a, b in cols]


@dataclass
class Preprocessing(BaseEstimator, TransformerMixin):
    """
    Load, merge, clean, and optionally pickle intermediate dataframe.
    Compatible with scikit-learn pipelines.

    Parameters
    ----------
    data_path : Union[str, Dict[str, str]]
        Directory with the two CSVs or dict {'energy': path, 'weather': path}.
    ffill_amt : Optional[int]
        Forward-fill limit (consecutive NaNs to fill). 0/None disables ffill.
    sampling_interval : int
        Sampling in hours. 1 means hourly.
    drop_na : bool
        Drop rows with any NaNs after filling/resampling.
    load_pickle_instead : Optional[str]
        If provided and exists, load that pickle and skip processing.
    save_pickle : bool
        Save processed dataframe next to energy file (or to provided pickle path).
    max_interp_hours : int
        Only interpolate *inside* gaps up to this many hours; larger gaps stay NaN
        to avoid deceptive long ramps across missing chunks.
    """
    data_path: Union[str, Dict[str, str]]
    ffill_amt: Optional[int] = 3
    sampling_interval: int = 1
    drop_na: bool = False
    load_pickle_instead: Optional[str] = None
    save_pickle: bool = True
    max_interp_hours: int = 6  # conservative default
    average_weather_data: bool = True
    drop_energy_details: bool = True

    def __post_init__(self):
        if isinstance(self.data_path, dict):
            if not all(k in self.data_path for k in ("energy", "weather")):
                raise ValueError("data_path dict must have keys {'energy','weather'}.")
        elif not isinstance(self.data_path, str):
            raise ValueError("data_path must be a directory path or a dict of file paths.")

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        # Load-from-pickle fast path
        if self.load_pickle_instead and os.path.exists(self.load_pickle_instead):
            print(f"[Preprocessing] Loading dataframe from pickle: {self.load_pickle_instead}")
            with open(self.load_pickle_instead, "rb") as f:
                df = pickle.load(f)
            return df

        # Resolve paths
        if isinstance(self.data_path, dict):
            energy_path = self.data_path["energy"]
            weather_path = self.data_path["weather"]
        else:
            energy_path = os.path.join(self.data_path, "energy_dataset.csv")
            weather_path = os.path.join(self.data_path, "weather_features.csv")
            if not os.path.exists(energy_path):
                c = [f for f in os.listdir(self.data_path) if "energy" in f and f.endswith(".csv")]
                if c:
                    energy_path = os.path.join(self.data_path, c[0])
            if not os.path.exists(weather_path):
                c = [f for f in os.listdir(self.data_path) if "weather" in f and f.endswith(".csv")]
                if c:
                    weather_path = os.path.join(self.data_path, c[0])

        if not os.path.exists(energy_path) or not os.path.exists(weather_path):
            raise FileNotFoundError(f"Could not find CSVs. energy: {energy_path}, weather: {weather_path}")

        energy = _load_csv(energy_path)
        weather = _load_csv(weather_path)

        # Drop useless cols from energy
        drop_energy_cols = ["generation hydro pumped storage aggregated",
                            "generation fossil coal-derived gas", "generation fossil oil shale",
                            "generation fossil peat", "generation geothermal", "generation marine",
                            "generation wind offshore", "forecast wind onshore day ahead", "forecast solar day ahead"]
        energy = energy.drop(columns=[c for c in drop_energy_cols if c in energy.columns])

        print(energy.columns)

        # Pivot weather by location if 'city_name' is present
        loc_candidates = ["city_name"]
        loc_map = {c.lower(): c for c in weather.columns}
        loc_col_key = next((k for k in loc_candidates if k in loc_map), None)
        if loc_col_key is not None:
            loc_col = loc_map[loc_col_key]
            num_cols = weather.select_dtypes(include=["number"]).columns.tolist()
            if loc_col in num_cols:
                num_cols.remove(loc_col)

            if self.average_weather_data:
                # Average numeric weather columns across all locations per timestamp
                weather = weather.groupby(weather.index)[num_cols].mean()
            else:
                # Pivot weather by location (existing behavior)
                wgrp = weather.groupby([weather.index, weather[loc_col]])[num_cols].mean()
                weather = wgrp.unstack(level=1)
                weather.columns = _flatten_columns(weather.columns)

        # De-duplicate timestamps by averaging
        if not energy.index.is_unique:
            energy = energy.groupby(energy.index).mean()
        if not weather.index.is_unique:
            weather = weather.groupby(weather.index).mean()

        # Drop weather cols if present
        drop_cols = ["weather_id", "weather_main", "weather_description", "weather_icon", "rain_3h", "snow_3h"]
        weather = weather.drop(columns=[c for c in drop_cols if c in weather.columns])

        # *** NO REINDEXING, NO INTERPOLATION ***
        # Align to the true intersection of timestamps
        common_idx = energy.index.intersection(weather.index).sort_values()
        energy = energy.loc[common_idx]
        weather = weather.loc[common_idx]

        # Merge (inner join is redundant now but explicit)
        df = energy.join(weather, how="inner", rsuffix="_weather")

        # Can patch small gaps
        if self.ffill_amt not in (None, 0):
            df = df.ffill(limit=int(self.ffill_amt))

        # always drop completely empty columns
        df = df.dropna(axis=1, how='all')

        if self.drop_na:
            df = df.dropna(axis=0, how="any")

        # Drop tz info for downstream models (keep local time correctness)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df.drop(columns=['time'])
        
        gen_cols = [c for c in df.columns if "generation" in c.lower()]
        
        if gen_cols:
            print("Processing generation columns for renewable/non-renewable totals...")
            lc = {c: c.lower() for c in gen_cols}
            renewable_keywords = ("wind", "solar", "hydro", "geothermal", "biomass", "marine", "renewable", "nuclear")
            nonrenewable_keywords = ("fossil", "coal", "gas", "oil", "peat", "waste")
            print(gen_cols)
            renew_cols = [c for c in gen_cols if any(k in lc[c] for k in renewable_keywords)]
            fossil_cols = [c for c in gen_cols if any(k in lc[c] for k in nonrenewable_keywords)]

            # Any generation columns not matched above are treated as non-renewable
            unmatched = [c for c in gen_cols if c not in renew_cols and c not in fossil_cols]
            if unmatched:
                fossil_cols += unmatched

            print(f"Identified renewable generation columns: {renew_cols}")
            print(df[renew_cols].sum(axis=1) if renew_cols else 0.0)
            
            df["total_renewable_generation"] = df[renew_cols].sum(axis=1) if renew_cols else 0.0
            df["total_fossil_generation"] = df[fossil_cols].sum(axis=1) if fossil_cols else 0.0
            df["renewable_generation_ratio"] = df["total_renewable_generation"] / (df["total_renewable_generation"] + df["total_fossil_generation"])

            # remove the original detailed generation columns
            if self.drop_energy_details:
                df = df.drop(columns=gen_cols)

        if self.save_pickle:
            pkl_path = self.load_pickle_instead or os.path.join(
                os.path.dirname(energy_path), "processed_dataframe.pkl"
            )
            try:
                df.to_csv(pkl_path.replace(".pkl", ".csv"))
                
                with open(pkl_path, "wb") as f:
                    pickle.dump(df, f)
                print(f"[Preprocessing] Saved processed dataframe to: {pkl_path}")
            except Exception as e:
                print(f"[Preprocessing] Could not save pickle ({e}). Skipping.")

        print('cols after preprocessing')
        for col in df.columns:
            print(col)



        return df
