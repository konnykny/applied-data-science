import numpy as np
import pycatch22
from typing import List, Tuple, Callable, Dict, Any
import pandas as pd
from joblib import delayed, Parallel
#from geopy.geocoders import Nominatim
import pvlib
from pvlib.location import Location

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, is_regressor, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class Catch22FeatureExtractor(BaseEstimator, TransformerMixin):


    def __init__(self, target_cols: List[str], windows_past_hrs: List[int], windows_future_hrs: List[int], njobs_default: int = 4) -> None:

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
        X_base = X.copy().iloc[start_idx : end_idx + 1]

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
                results_col = Parallel(n_jobs=njobs)(delayed(pycatch22.catch22_all)(X[col].iloc[start_ids[i]:end_ids[i]+1], catch24=True) for i in range(start_ids.shape[0]))
                values = np.concatenate([np.asarray(v['values'], dtype=float).reshape(-1, 1) for v in results_col], axis=1)
                names = results_col[0]['names']
                # rename
                names = [f'catch22_{self.windows_past_hrs[k]}_{self.windows_future_hrs[k]}__{col} {name}' for name in names]
                for name, value in zip(names, values):
                    result_dict[name] = np.asarray(value)
        
        result_df = pd.DataFrame(result_dict).reset_index(drop=True)

        # join
        result_df = pd.concat([X_base, result_df], axis=1)

        return result_df
    
    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None, njobs: int = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _precompute_window_ids(self, src_time_col: pd.Series, query_time_col: pd.Series) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
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
        #geolocator = Nominatim(user_agent='demo')
        #self.city_lat_lon = {}
        #for city in self.cities:
        #    location = geolocator.geocode(query=city)
        #    assert location is not None
        #    city_lat, city_lon = location.latitude, location.longitude
        #    self.city_lat_lon[city] = (city_lat, city_lon)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> 'WeatherFeaturesExtractor':
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.copy()
        sun_features = self.compute_sun_features(X)
        wind_features = self.compute_wind_deg_interactions(X)
        X = pd.concat([X, wind_features, sun_features], axis=1)
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
        
    def compute_wind_deg_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        
        result_df = pd.DataFrame()
        for city in self.cities:
            result_df[f'{city} wind_sin_speed'] = X[f'{city} wind_speed'] * np.sin(np.deg2rad(X[f'{city} wind_deg']))
            result_df[f'{city} wind_cos_speed'] = X[f'{city} wind_speed'] * np.cos(np.deg2rad(X[f'{city} wind_deg']))
        assert pd.isna(result_df).values.sum() == 0    
        return result_df

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
            'daylight_hours': loc.get_sun_rise_set_transit(dt_index).apply(lambda x: (x['sunset'] - x['sunrise']).total_seconds()/3600, axis=1)
        })
        features = features.reset_index(drop=True)

        return features
    

class CyclicFeaturesExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, target_cols: List[str], max_vals: List[float]) -> None:
        super().__init__()

        assert len(max_vals) == len(target_cols)
        self.target_cols = target_cols
        self.max_vals = max_vals

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> 'WeatherFeaturesExtractor':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.copy()
        for col, max_val in zip(self.target_cols, self.max_vals):
            X[f"{col}_sin"] = np.sin(2 * np.pi * X[col] / max_val)
            X[f"{col}_cos"] = np.cos(2 * np.pi * X[col] / max_val)
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
        


    
# NOTE: ChatGPT
class DFColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies different transformations to specified column groups
    of a pandas DataFrame, preserving original column names and supporting passthrough.
    """
    def __init__(self, transformers, remainder='drop'):
        """
        Parameters:
        -----------
        transformers : list of tuples
            List of (name, transformer, columns) tuples specifying the transformer objects
            to be applied to subsets of columns.
        remainder : {'drop', 'passthrough'} or transformer, default='drop'
            By default, only specified columns are transformed, and the rest are dropped.
            If 'passthrough', remaining columns are included unchanged.
            If a transformer, it is applied to remaining columns.
        """
        self.transformers = transformers
        self.remainder = remainder

    def fit(self, X, y=None):
        """Fit all transformers using X."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.input_columns_ = X.columns.tolist()
        self.transformers_ = []
        self.remainder_columns_ = []
        self.output_columns_ = []

        # Track used columns to determine remainder
        used_columns = set()

        # Fit each transformer
        for name, transformer, columns in self.transformers:
            if columns is None or len(columns) == 0:
                continue
            if isinstance(columns, str):
                columns = [columns]
            if not all(c in self.input_columns_ for c in columns):
                missing = [c for c in columns if c not in self.input_columns_]
                raise ValueError(f"Columns {missing} not found in input DataFrame")
            used_columns.update(columns)
            # Clone and fit transformer
            from sklearn.base import clone
            trans = clone(transformer)
            trans.fit(X[columns], y)
            self.transformers_.append((name, trans, columns))

        # Handle remainder
        if self.remainder != 'drop':
            self.remainder_columns_ = [c for c in self.input_columns_ if c not in used_columns]
            if self.remainder != 'passthrough':
                self.remainder_transformer_ = clone(self.remainder)
                self.remainder_transformer_.fit(X[self.remainder_columns_], y)
            else:
                self.remainder_transformer_ = None

        # Build output column names (preliminary)
        self._build_output_columns()
        return self

    def transform(self, X):
        """Transform X by applying transformers to specified columns."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        check_is_fitted(self, 'transformers_')

        if list(X.columns) != self.input_columns_:
            raise ValueError("Input DataFrame columns do not match those seen during fit")

        # Collect transformed data and names dynamically
        transformed_data = []
        names = []

        # Apply each transformer
        for name, transformer, columns in self.transformers_:
            X_subset = transformer.transform(X[columns])
            # Get names preferably from output if DataFrame
            if isinstance(X_subset, pd.DataFrame):
                sub_names = X_subset.columns.tolist()
                X_subset = X_subset.values
            else:
                if hasattr(transformer, 'get_feature_names_out'):
                    sub_names = transformer.get_feature_names_out()
                else:
                    sub_names = columns
            # Check if matches output shape
            n_out = X_subset.shape[1] if X_subset.ndim == 2 else 1
            if len(sub_names) != n_out:
                sub_names = [f"{name}_{columns[0]}_f{j}" for j in range(n_out)]  # Generic if mismatch
            names.extend(sub_names)
            transformed_data.append(X_subset if X_subset.ndim == 2 else X_subset.reshape(-1, 1))

        # Handle remainder
        if self.remainder != 'drop':
            if self.remainder == 'passthrough':
                X_remainder = X[self.remainder_columns_].values
                rem_names = self.remainder_columns_
            else:
                X_remainder = self.remainder_transformer_.transform(X[self.remainder_columns_])
                if isinstance(X_remainder, pd.DataFrame):
                    rem_names = X_remainder.columns.tolist()
                    X_remainder = X_remainder.values
                else:
                    if hasattr(self.remainder_transformer_, 'get_feature_names_out'):
                        rem_names = self.remainder_transformer_.get_feature_names_out()
                    else:
                        rem_names = self.remainder_columns_
            # Check if matches
            n_out = X_remainder.shape[1] if X_remainder.ndim == 2 else 1
            if len(rem_names) != n_out:
                rem_names = [f"remainder_f{j}" for j in range(n_out)]
            names.extend(rem_names)
            transformed_data.append(X_remainder if X_remainder.ndim == 2 else X_remainder.reshape(-1, 1))

        # Concatenate all transformed data
        if not transformed_data:
            raise ValueError("No data to transform")
        X_transformed = np.hstack(transformed_data)

        # Debugging: Uncomment to check
        #print(f"Transformed shape: {X_transformed.shape}")
        #print(f"Number of column names: {len(names)}")
        #print(f"Column names: {names}")

        # Create DataFrame
        return pd.DataFrame(X_transformed, columns=names, index=X.index)

    def fit_transform(self, X, y=None):
        """Fit and transform X."""
        return self.fit(X, y).transform(X)

    def _build_output_columns(self):
        """Build the list of output column names for get_feature_names_out."""
        self.output_columns_ = []
        for name, transformer, columns in self.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out()
            else:
                names = columns
            self.output_columns_.extend(names)
        if self.remainder != 'drop':
            self.output_columns_.extend(self.remainder_columns_)

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        check_is_fitted(self, 'output_columns_')
        return self.output_columns_