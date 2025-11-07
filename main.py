#%%
import os
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import Preprocessing
from exploration_pipe import Exploration
from features import FeatureExtraction, SaveORLoad
from modeling import ModelTraining, MultiRegXGBoostTraining


def build_pipeline(data_dir_or_paths, target_column=None):

    pipe = Pipeline(steps=[
        ("#1 Preprocessing", Preprocessing(
            data_path = data_dir_or_paths,
            ffill_amt = 3,
            sampling_interval = 1,
            drop_na = True,
            load_pickle_instead = None,# './data/preprocessed_dataframe.pkl',
            save_pickle = False,
            drop_energy_details = True,
        )),
        ("#2 Explore", Exploration(
            nan_report=True,
            plot_data= 'weather',  # "All" | "weather" | "energy" | None | "seaborn"
        )),
        ("#3 Calculate features", FeatureExtraction(
            target_column,
            N_past_values= [24*2], ###
            N_future_values= 24,
            N_past_target_values=24,
            weather_columns= ['wind_speed', 'madrid solar_azimuth', 'madrid dni'],
            generated_locations = ['madrid'], # choose city locations for feature generation (zenith, azimuth, etc.)
            future_columns = [target_column, 'temp', 'wind_speed', 'rain_1h', 'madrid solar_azimuth', 'madrid dni'],
            prediction_time_of_day = 0, # local time hour (24hrs) for which prediction should be trained and later run
            use_pickle = False,
            add_raw_target=True,
        )),
        ("#3.5 Explore", Exploration(
            nan_report=True,
            plot_data=None,  # "All" | "weather" | "energy" | None | "seaborn"
        )),
        ("save_features", SaveORLoad(mode='save')),
        ("load_features", SaveORLoad(mode='load')),
        ("#4 Train Models", MultiRegXGBoostTraining( #ModelTraining
            target_column = 'green_generation_ratio',
            test_size=0.2,
            random_state=42,
            plot_results=True,
            prediction_hour=0
        ))
    ])
    return pipe


pipeline = build_pipeline(
    data_dir_or_paths='./data', #os.environ.get("DATA_DIR", "."),
    target_column='green_generation_ratio'
)
prediction = pipeline.fit_transform(None)

#%%

print(prediction) # TODO! use this as input for optimisation part

pass # for breakpoint
print("[main] Pipeline completed.")


#%%

### really ugly plotting code for prediction results inspection

week_num = 9
target = 'green_generation_ratio'

df = prediction.copy() if hasattr(prediction, 'copy') else pd.DataFrame(prediction)
# prefer original tz-aware time column
if 'time__original_tz' in df.columns:
    df.index = pd.to_datetime(df['time__original_tz'], utc=True)
elif 'timestamp' in df.columns:
    df.index = pd.to_datetime(df['timestamp'], utc=True)
else:
    df.index = pd.to_datetime(df.index, errors='coerce')

import re

# collect columns grouped by trailing horizon integer
cols = list(df.columns)
h_groups = {}
for c in cols:
    m = re.search(r'(\d{1,3})\)?$', c)
    if not m:
        continue
    h = int(m.group(1))
    h_groups.setdefault(h, []).append(c)

# classify each group's columns into pred vs gt (heuristics)
pred_parts = []
gt_parts = []
for h, group in sorted(h_groups.items()):
    pred_col = None
    gt_col = None
    for c in group:
        lc = c.lower()
        if any(k in lc for k in ('pred', 'forecast', 'prediction')):
            pred_col = c
        if any(k in lc for k in ('ground', 'gt', 'true')):
            gt_col = c
    # if target appears in a column name and no explicit pred keyword, treat as gt
    if not gt_col:
        for c in group:
            if target.lower() in c.lower() and not any(k in c.lower() for k in ('pred', 'forecast', 'prediction')):
                gt_col = c
                break
    # fallback: if neither explicit, assume first is pred
    if not pred_col and not gt_col and group:
        pred_col = group[0]
    if pred_col:
        pred_parts.append((h, pred_col))
    if gt_col:
        gt_parts.append((h, gt_col))

# expand lists into hourly Series
def _expand(df, parts):
    series_parts = []
    for h, col in parts:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        s.index = pd.to_datetime(s.index)
        series_parts.append(pd.Series(s.values, index=s.index + pd.to_timedelta(int(h), unit='h')))
    if not series_parts:
        return pd.Series(dtype=float)
    return pd.concat(series_parts).groupby(level=0).mean().sort_index()

pred_s = _expand(df, pred_parts)
gt_s = _expand(df, gt_parts)

# single-column fallback if expansion found nothing
if pred_s.empty and gt_s.empty:
    pred_col = next((c for c in [f'prediction_{target}', f'pred_{target}', f'{target}_pred', target] if c in df.columns), None)
    gt_col = next((c for c in [f'ground_truth_{target}', f'{target}_gt', f'gt_{target}'] if c in df.columns), None)
    # build hourly index for the chosen week
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.isna().all():
        base = df.index.min()
    else:
        base = pd.to_datetime('1970-01-01')
    week_start = base + pd.Timedelta(weeks=week_num)
    hours = pd.date_range(start=week_start, periods=168, freq='H', tz=getattr(df.index, 'tz', None))
    dfw = df.reindex(hours)
    plt.figure(figsize=(14, 4))
    shown = False
    if pred_col and pred_col in dfw and not dfw[pred_col].dropna().empty:
        plt.plot(hours, dfw[pred_col].values, label='prediction'); shown = True
    if gt_col and gt_col in dfw and not dfw[gt_col].dropna().empty:
        plt.plot(hours, dfw[gt_col].values, label='ground_truth'); shown = True
    if not shown:
        print('[plot] no pred/gt columns found for the selected week; available columns:', list(df.columns))
    else:
        plt.legend(); plt.title(f'Hourly — week {week_num} — {target}'); plt.tight_layout(); plt.show()
else:
    # reindex expanded series to exact 168 hourly points and plot both
    starts = [s.index.min() for s in (pred_s, gt_s) if not s.empty]
    base = min(starts) if starts else pd.to_datetime('1970-01-01')
    week_start = base + pd.Timedelta(weeks=week_num)
    hours = pd.date_range(start=week_start, periods=168, freq='H', tz=getattr(base, 'tz', None))
    p = pred_s.reindex(hours)
    g = gt_s.reindex(hours)
    plt.figure(figsize=(14, 4))
    shown = False
    if not p.dropna().empty:
        plt.plot(hours, p.values, label='prediction'); shown = True
    if not g.dropna().empty:
        plt.plot(hours, g.values, label='ground_truth'); shown = True
    if not shown:
        print(f'[plot] no hourly pred/gt data for week starting {week_start} (week_num={week_num}).')
    else:
        plt.legend(); plt.title(f'Hourly predictions vs ground_truth — week {week_num} — {target}'); plt.tight_layout(); plt.show()
