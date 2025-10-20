import os
import os.path as osp
import pandas as pd
import argparse
import holidays


WEATHER_COLS = ['temp', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all', 'weather_id', 'weather_main', 'weather_description', 'weather_icon']
RENEWABLE_GENERATION_COLS = ['generation biomass', 'generation geothermal', 'generation hydro run-of-river and poundage', 'generation hydro water reservoir', 'generation marine', 'generation other renewable', 'generation solar', 'generation wind offshore', 'generation wind onshore', 'generation nuclear']  # NOTE: Nuclear renewable?
GENERATION_COLS = RENEWABLE_GENERATION_COLS + ['generation fossil brown coal/lignite', 'generation fossil coal-derived gas', 'generation fossil gas', 'generation fossil hard coal', 'generation fossil oil', 'generation fossil oil shale', 'generation fossil peat']

def merge_dataset(energy_df: pd.DataFrame, weather_df: pd.DataFrame, fill_hrs: int = None) -> pd.DataFrame:
    out_df = energy_df.copy()

    # initialize weather features per city
    city_names = weather_df['city_name'].unique()
    for city_name in city_names:
        for col in WEATHER_COLS:
            out_df[f"{city_name.lower().strip()} {col}"] = pd.Series(dtype=weather_df[col].dtype)
    
    # fill the initialized columns with values
    for city_name in city_names:
        city_weather_df = weather_df[weather_df['city_name'] == city_name]
        for col in WEATHER_COLS:
            out_df.loc[out_df['time'].isin(city_weather_df['dt_iso']), f'{city_name.lower().strip()} {col}'] = out_df['time'].map(dict(zip(city_weather_df['dt_iso'], city_weather_df[col])))

    # drop all nan columns
    out_df = out_df.dropna(axis=1, how='all')

    # ensire datetime format and indexing
    out_df['time'] = pd.to_datetime(out_df['time'], utc=True, errors='coerce')
    out_df.set_index('time', inplace=True)
    out_df = out_df.sort_index()
    assert isinstance(out_df.index, pd.DatetimeIndex)

    if fill_hrs is not None:
        # fill missing data with median of past `fill_hrs` hours (for numerics)
        for col_name in out_df.columns:
            if pd.api.types.is_numeric_dtype(out_df[col_name]):
                filling_rolling_median = out_df[col_name].rolling(f'{fill_hrs}H', min_periods=1).median()
                out_df[col_name] = out_df[col_name].fillna(filling_rolling_median).fillna(method='ffill').interpolate(method='nearest', limit_direction='both')
        
        # fill non-numeric columns (e.g., categorical weather) with ffill then bfill
        for col_name in out_df.columns:
            if out_df[col_name].dtype == 'object':
                out_df[col_name] = out_df[col_name].fillna(method='ffill').fillna(method='bfill')  # TODO()

    # compute total generation and renewable generation - NOTE: includes nuclear in renewables
    out_df['total generation'] = out_df[GENERATION_COLS].sum(axis=1)
    out_df['renewable generation'] = out_df[RENEWABLE_GENERATION_COLS].sum(axis=1)
    out_df['renewable generation percent'] = out_df['renewable generation'] / out_df['total generation']

    # add time-based features
    out_df['day of week'] = out_df.index.dayofweek
    dt_series = out_df.index.to_series()
    es_holidays = holidays.ES(years=dt_series.dt.year.min())  # Efficient: specify year range
    out_df['vacation'] = dt_series.apply(
        lambda x: es_holidays.get(x, 'no vacation').lower()  # Returns holiday name (e.g., 'New Year') or 'None'
    )
    out_df['hour'] = dt_series.dt.hour
    out_df['month'] = dt_series.dt.month
    out_df['day'] = dt_series.dt.day


    return out_df


def merge_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy_path",
        type=str,
        default="data/energy_dataset.csv",
        help="Path to the energy data CSV file.",
    )
    parser.add_argument(
        "--weather_path",
        type=str,
        default="data/weather_features.csv",
        help="Path to the weather data CSV file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/merged_data.csv",
        help="Path to save the merged dataset CSV file.",
    )
    parser.add_argument(
        "--fill_hrs",
        type=int,
        default=None,
        help="Number of hours to compute the median to fill missing weather data.",
    )
    args = parser.parse_args()
    assert osp.isfile(args.energy_path), f"Energy data file not found: {args.energy_path}"
    assert osp.isfile(args.weather_path), f"Weather data file not found: {args.weather_path}"
    assert osp.isdir(osp.dirname(args.output_path)), f"Output directory not found: {osp.dirname(args.output_path)}"


    energy_pd = pd.read_csv(args.energy_path)
    weather_pd = pd.read_csv(args.weather_path)

    merged_pd = merge_dataset(energy_pd, weather_pd, fill_hrs=args.fill_hrs)

    merged_pd.to_csv(args.output_path, index=True)
    print(f"Merged dataset saved to {args.output_path}")    # Merge the two datasets on the 'timestamp' column
    


if __name__ == "__main__":
    merge_main()
    print('done')

