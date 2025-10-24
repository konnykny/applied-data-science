
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from joblib import delayed, Parallel
import pvlib
from pvlib.location import Location
import pycatch22

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _cyclic_encode(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    x = 2 * np.pi * series / period
    return pd.DataFrame({
        f"{prefix}_sin": np.sin(x),
        f"{prefix}_cos": np.cos(x),
    }, index=series.index)


def _infer_energy_groups(columns: pd.Index) -> Dict[str, List[str]]:
    renew_kw = ["solar", "wind", "hydro", "biomass", "geothermal", "renew"]
    nonrenew_kw = ["coal", "gas", "oil", "nuclear", "lignite", "peat", "non_renew"]
    usage_kw = ["load", "consumption", "demand", "usage"]

    def match(keys):
        return [c for c in columns if any(k in c.lower() for k in keys)]

    return {
        "renewable": match(renew_kw),
        "non_renewable": match(nonrenew_kw),
        "usage": match(usage_kw),
    }


def _rolling_features(x: pd.Series, window: int, prefix: str) -> pd.DataFrame:
    r = x.rolling(window=window, min_periods=max(1, window // 3))
    df = pd.DataFrame({
        f"{prefix}_roll_mean": r.mean(),
        f"{prefix}_roll_std": r.std(),
        f"{prefix}_roll_min": r.min(),
        f"{prefix}_roll_max": r.max(),
        f"{prefix}_lag1": x.shift(1),
    })
    return df


def _catch22_features(x: pd.Series, window: int, prefix: str) -> pd.DataFrame:
    try:
        from pycatch22 import catch22_all
    except Exception:
        return _rolling_features(x, window, prefix)

    out = []
    names = None
    values = x.values
    for i in range(len(x)):
        start = max(0, i - window + 1)
        segment = values[start:i+1]
        if np.isnan(segment).all():
            out.append(None)
            continue
        try:
            res = catch22_all(np.asarray(segment, dtype=float))
            if names is None:
                names = [f"{prefix}_c22_{n}" for n in res["names"]]
            out.append(res["values"])
        except Exception:
            out.append(None)

    if not names:
        return _rolling_features(x, window, prefix)

    arr = np.full((len(x), len(names)), np.nan, dtype=float)
    for i, vals in enumerate(out):
        if vals is not None:
            arr[i, :] = vals
    df = pd.DataFrame(arr, index=x.index, columns=names)
    df[f"{prefix}_lag1"] = x.shift(1)
    return df


# features
#
#

@dataclass
class FeatureExtraction(BaseEstimator, TransformerMixin):
    N_past_values: int = 72

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureExtraction expects a pandas DataFrame.")

        df = X.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("FeatureExtraction requires a DatetimeIndex.")

        groups = _infer_energy_groups(df.columns)
        feats = {}
        for name, cols in groups.items():
            if cols:
                feats[f"sum_{name}"] = df[cols].sum(axis=1)

        weekday = _cyclic_encode(pd.Series(df.index.weekday, index=df.index), 7, "weekday")
        minute_of_day = df.index.hour * 60 + df.index.minute
        time_of_day = _cyclic_encode(pd.Series(minute_of_day, index=df.index), 24 * 60, "tod")
        day_of_year = _cyclic_encode(pd.Series(df.index.dayofyear, index=df.index), 366, "doy")

        # TODO add one-hot encoding for holidays/vacations

        feature_frames = [weekday, time_of_day, day_of_year]

        window = int(self.N_past_values)
        target_cols = list(feats.keys())
        energy_cols_all = set(sum(groups.values(), []))
        weather_cols = [c for c in df.columns if c not in energy_cols_all]

        for col in target_cols:
            feature_frames.append(_catch22_features(feats[col], window, prefix=col))

        for col in weather_cols:
            if np.issubdtype(df[col].dtype, np.number):
                feature_frames.append(_catch22_features(df[col], window, prefix=f"w_{col}"))

        F = pd.concat(feature_frames, axis=1)
        F = F.dropna(axis=1, how="all")

        F.to_csv("features_extracted.csv")

        return F


#####################

class Catch22FeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, target_cols: List[str], windows_past_hrs: List[int], windows_future_hrs: List[int],
                 njobs_default: int = 4) -> None:

        assert len(windows_future_hrs) == len(windows_past_hrs)
        for k_p, k_f in zip(windows_past_hrs, windows_future_hrs):
            assert k_p + k_f >= 3

        self.target_cols = target_cols
        self.njobs_default = njobs_default

        # incides to crop the merged dataframe
        self.windows_past_hrs = windows_past_hrs
        self.windows_future_hrs = windows_future_hrs

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> 'Catch22FeatureExtractor':

        return self

    def transform(self, X: pd.DataFrame, njobs: int = None) -> pd.DataFrame:

        # sort X by time
        X = X.copy()
        X['time'] = pd.to_datetime(X['time'])
        X = X.sort_values('time', ascending=True)
        X = X.reset_index(drop=True)

        assert pd.isna(X[self.target_cols]).values.sum() == 0

        # crop the base X dataframe to fit the extracted features
        start_idx = np.max(self.windows_past_hrs)
        end_idx = X.shape[0] - np.max(self.windows_future_hrs) - 1
        X_base = X.copy().iloc[start_idx: end_idx + 1]

        njobs = self.njobs_default if njobs is None else njobs

        # extract the indices
        src_time_col = X['time']
        query_time_col = X_base['time']
        start_ids_list, end_ids_list, valid_mask_list = self._precompute_window_ids(src_time_col, query_time_col)

        # merge the mask and reducte the query
        combined_mask = valid_mask_list[0]
        for mask in valid_mask_list[1:]:
            combined_mask &= mask

        X_base = X_base[combined_mask].reset_index(drop=True)
        start_ids_list = [start_ids[combined_mask] for start_ids in start_ids_list]
        end_ids_list = [end_ids[combined_mask] for end_ids in end_ids_list]

        # compute all features per window
        assert pd.isna(X).values.sum() == 0
        result_dict = {}
        for col in self.target_cols:
            for k, bounds in enumerate(zip(start_ids_list, end_ids_list)):
                start_ids, end_ids = bounds
                results_col = Parallel(n_jobs=njobs)(
                    delayed(pycatch22.catch22_all)(X[col].iloc[start_ids[i]:end_ids[i] + 1], catch24=True) for i in
                    range(start_ids.shape[0]))
                values = np.concatenate([np.asarray(v['values'], dtype=float).reshape(-1, 1) for v in results_col],
                                        axis=1)
                names = results_col[0]['names']
                # rename
                names = [f'catch22_{self.windows_past_hrs[k]}_{self.windows_future_hrs[k]}__{col} {name}' for name in
                         names]
                for name, value in zip(names, values):
                    result_dict[name] = np.asarray(value)

        result_df = pd.DataFrame(result_dict).reset_index(drop=True)

        # join
        result_df = pd.concat([X_base, result_df], axis=1)

        return result_df

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None, njobs: int = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _precompute_window_ids(self, src_time_col: pd.Series, query_time_col: pd.Series) -> Tuple[
        List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        # assumes a sorted column by time (ascending)

        src_time_index = pd.DatetimeIndex(src_time_col)

        future_offsets = pd.to_timedelta(np.asarray(self.windows_future_hrs), unit='h')
        past_offsets = pd.to_timedelta(np.asarray(self.windows_past_hrs), unit='h')

        start_ids_list = []
        end_ids_list = []
        valid_mask_list = []

        for p_hrs, f_hrs in zip(past_offsets, future_offsets):
            query_start_times = query_time_col - p_hrs
            query_end_times = query_time_col + f_hrs

            start_ids = src_time_index.get_indexer(pd.DatetimeIndex(query_start_times))
            end_ids = src_time_index.get_indexer(pd.DatetimeIndex(query_end_times))
            valid_mask = (start_ids != -1) & (end_ids != -1)

            start_ids_list.append(start_ids)
            end_ids_list.append(end_ids)
            valid_mask_list.append(valid_mask)

        return start_ids_list, end_ids_list, valid_mask_list


####### Weather



class WeatherFeaturesExtractor(BaseEstimator, TransformerMixin):

    city_lat_lon = {
        'seville': (37.3886, -5.9823),
        'barcelona': (41.3825, 2.1769),
        'madrid': (40.4165, -3.7026),
        'bilbao': (43.2630, -2.9349),
        'valencia': (39.4699, -0.3763)
    }

    def __init__(self, cities: List[str] = ['seville', 'barcelona', 'madrid', 'bilbao', 'valencia']) -> None:

        assert len(cities) > 0

        self.cities = cities
        # geolocator = Nominatim(user_agent='demo')
        # self.city_lat_lon = {}
        # for city in self.cities:
        #    location = geolocator.geocode(query=city)
        #    assert location is not None
        #    city_lat, city_lon = location.latitude, location.longitude
        #    self.city_lat_lon[city] = (city_lat, city_lon)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> 'WeatherFeaturesExtractor':

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.copy()
        sun_features = self.compute_sun_features(X)
        X = pd.concat([X, sun_features], axis=1)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def compute_sun_features(self, X: pd.DataFrame) -> pd.DataFrame:

        fea_dfs_list = []
        for city in self.cities:
            city_lat, city_lon = self.city_lat_lon[city]
            # get the solar featiures
            city_fea_df = self.generate_pv_wind_features(lat=city_lat, lon=city_lon, datetimes=pd.to_datetime(X['time'].copy()))
            # reindex if nescessary
            if not city_fea_df.index.equals(X.index):
                city_fea_df = city_fea_df.reindex(X.index)
            # rename the featires uniquely for the city
            city_fea_df.columns = [f'{city} {col}' for col in city_fea_df.columns]
            fea_dfs_list.append(city_fea_df)

        # concat
        result_df = pd.concat(fea_dfs_list, axis=1)
        assert pd.isna(result_df).values.sum() == 0
        return result_df

    def generate_pv_wind_features(self, lat: float, lon: float, datetimes: pd.Series) -> pd.DataFrame:
        dt_index = pd.DatetimeIndex(pd.to_datetime(datetimes)).tz_convert('Europe/Madrid')
        loc = Location(latitude=lat, longitude=lon, tz='Europe/Madrid')
        solpos = loc.get_solarposition(dt_index)
        clearsky = loc.get_clearsky(dt_index)
        features = pd.DataFrame({
            'solar_zenith': solpos['zenith'].values,
            'solar_azimuth': solpos['azimuth'].values,
            'ghi': clearsky['ghi'].values,
            'dni': clearsky['dni'].values,
            'dhi': clearsky['dhi'].values,
            'daylight_hours': loc.get_sun_rise_set_transit(dt_index).apply(
                lambda x: (x['sunset'] - x['sunrise']).total_seconds() / 3600, axis=1)
        })
        features = features.reset_index(drop=True)

        return features
    
    