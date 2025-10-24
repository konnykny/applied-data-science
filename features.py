
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from joblib import delayed, Parallel
import pvlib
from pvlib.location import Location
import pycatch22
# import tqdm

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _cyclic_encode(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    x = 2 * np.pi * series / period
    return pd.DataFrame({
        f"{prefix}_sin": np.sin(x),
        f"{prefix}_cos": np.cos(x),
    }, index=series.index)


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
# 3x cyclic encoded time features
# - weekday
# - time of day
# - day of year -> NOT every hour but only once in the prediction

# c22 weather features for previous day (pa2st 4 hours)

# raw green percentage values for past N hours (control with N_future_values)

# c22 weather features for next 24 hours

### start prediction 2x a day from (local time) eg 8AM and again from eg 5 PM

class FeatureExtraction(BaseEstimator, TransformerMixin):
    
    def __init__(self, target_column:str, weather_columns: List[str], N_past_values: List[int] = [24, 7*24], N_future_values: int = 24,
                 n_jobs: int = 4, add_raw_target = True, N_past_target_values: int = 24, verbose = 1) -> None:

        self.target_column = target_column
        self.verbose = verbose
        self.weather_cols = weather_columns
        self.N_past_target_values = N_past_target_values
        self.weather_extractor = WeatherFeaturesExtractor()
        self.add_raw_target = add_raw_target
        # initialize with empty cols -> set before usage in transform
        all_windows_past = [*N_past_values, 0]
        all_windows_future = [*[0 for _ in N_past_values], N_future_values]
        self.catch22_weather_extractor = Catch22FeatureExtractor(target_cols=weather_columns, windows_past_hrs=all_windows_past, 
                                                                 windows_future_hrs=all_windows_future, njobs_default=n_jobs)
    
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> 'FeatureExtraction':
        if (self.target_column is None) or (self.target_column not in X.columns):
            raise ValueError("FeatureExtraction requires a target_column to be specified.")        
        else:
            return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureExtraction expects a pandas DataFrame.")

        df = X.copy()

        # make sure the dataframe is sorted
        # df = df.sort_values(by=['time'], ascending=True) # TODO!!

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("FeatureExtraction requires a DatetimeIndex.")

        weekday = _cyclic_encode(pd.Series(df.index.weekday, index=df.index), 7, "weekday")
        minute_of_day = df.index.hour * 60 + df.index.minute
        time_of_day = _cyclic_encode(pd.Series(minute_of_day, index=df.index), 24 * 60, "tod")
        day_of_year = _cyclic_encode(pd.Series(df.index.dayofyear, index=df.index), 366, "doy")

        # TODO add one-hot encoding for holidays/vacations

        feature_frames = [weekday, time_of_day, day_of_year]

        # compute the additional astronomical features
        # sun_features = self.weather_extractor.transform(df)
        # feature_frames.append(sun_features)        

        F = pd.concat(feature_frames, axis=1)
        F = pd.concat([df, F], axis=1)

        # compute the catch22 features and crop
        if self.verbose:
            print("Computing catch22 weather features...")
        F = self.catch22_weather_extractor.transform(F)

        # Add raw energy percentage columns
        if self.add_raw_target:
            # Add raw energy percentage columns (previous values)
            # create N previous-hour columns named f"{target_column}_{i}" where
            # i=0 corresponds to value at (current_time - N hours), i increases up to N-1 -> (current_time - 1 hour)
            N = getattr(self, "N_future_windows", None)
            if N is None:
                # fallback to the configured past-target-window size if explicit future-windows wasn't stored
                N = int(self.N_past_target_values)
            else:
                N = int(N)

            # Use the original df target series, reindexed to the (possibly cropped) F index returned by catch22
            orig_target = df[self.target_column].reindex(F.index)

            for i in range(N):
                # shift amount: i=0 -> shift N (value at t-N hours), i=N-1 -> shift 1 (value at t-1 hour)
                shift_amount = N - i
                col_name = f"{self.target_column}_{i}"
                F[col_name] = orig_target.shift(shift_amount)
            
        F = F.dropna(axis=1, how="all")
        
        return F


class SaveORLoad(BaseEstimator, TransformerMixin):
    def __init__(self, mode: str, data_path = '.\data', filename = 'features.csv') -> None:
        self.data_path = data_path
        self.filename = filename
        self.mode = mode  # 'load' or 'save'
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mode == 'load':
            X = pd.read_csv(f"{self.data_path}/{self.filename}")
            return X
        elif self.mode == 'save':
            X.to_csv(f"{self.data_path}/{self.filename}")
            return X

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
        # X['time'] = pd.to_datetime(X.index)
        # X = X.sort_values('time', ascending=True)
        X = X.reset_index(drop=True)

        assert pd.isna(X[self.target_cols]).values.sum() == 0

        # crop the base X dataframe to fit the extracted features
        start_idx = np.max(self.windows_past_hrs)
        end_idx = X.shape[0] - np.max(self.windows_future_hrs) - 1
        X_base = X.copy().iloc[start_idx: end_idx + 1]

        njobs = self.njobs_default if njobs is None else njobs

        # extract the indices
        src_time_col = X.index
        query_time_col = X_base.index
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
                names = [f'catch22_{self.windows_past_hrs[k]}_{self.windows_future_hrs[k]}__{col}_{name}' for name in
                         names]
                for name, value in zip(names, values):
                    result_dict[name] = np.asarray(value)

        result_df = pd.DataFrame(result_dict).reset_index(drop=True)
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
        
        return sun_features

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def compute_sun_features(self, X: pd.DataFrame) -> pd.DataFrame:

        fea_dfs_list = []
        for city in self.cities:
            city_lat, city_lon = self.city_lat_lon[city]
            # get the solar featiures
            city_fea_df = self.generate_pv_wind_features(lat=city_lat, lon=city_lon, df=X)
            # reindex if nescessary
            if not city_fea_df.index.equals(X.index):
                city_fea_df = city_fea_df.reindex(X.index)
            # rename the featires uniquely for the city
            city_fea_df.columns = [f'{city} {col}' for col in city_fea_df.columns]
            fea_dfs_list.append(city_fea_df)

        # concat
        result_df = pd.concat(fea_dfs_list, axis=1)
        print(result_df[result_df.isna()])
        # assert pd.isna(result_df).values.sum() == 0
        return result_df

    def generate_pv_wind_features(self, lat: float, lon: float, df) -> pd.DataFrame:
        # parse times robustly and ensure they are timezone-aware in Europe/Madrid
        dt_index = pd.DatetimeIndex(df.index, tz='UTC')
        
        # if naive, localize to Europe/Madrid; if already tz-aware, convert to Europe/Madrid


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
    
    