#%%
import os
from sklearn.pipeline import Pipeline
from preprocessing import Preprocessing
from exploration import Exploration
from features import FeatureExtraction, SaveORLoad
from modeling import ModelTraining, MultiRegXGBoostTraining


def build_pipeline(data_dir_or_paths, target_column=None):

    pipe = Pipeline(steps=[
        # ("#1 Preprocessing", Preprocessing(
        #     data_path = data_dir_or_paths,
        #     ffill_amt = 3,
        #     sampling_interval = 1,
        #     drop_na = True,
        #     load_pickle_instead = './data/preprocessed_dataframe.pkl',
        #     save_pickle = False,
        #     drop_energy_details = True,
        # )),
        # # ("#2 Explore", Exploration(
        # #     nan_report=True,
        # #     plot_data=None,  # "All" | "weather" | "energy" | None | "seaborn"
        # # )),
        # ("#3 Calculate features", FeatureExtraction(
        #     target_column,
        #     N_past_values= [24, 24*7],
        #     N_future_values= 24,
        #     N_past_target_values=24,
        #     weather_columns= ['temp', 'wind_speed', 'clouds_all', 'rain_1h'],
        #     generated_locations = ['madrid'], # choose city locations for feature generation (zenith, azimuth, etc.)
        #     future_columns = [target_column, 'temp', 'wind_speed', 'clouds_all', 'rain_1h'],
        #     prediction_time_of_day = 8, # local time hour when prediction should be trained and run
        #     use_pickle = False,
        #     add_raw_target=True,
        # )),
        # ("#3.5 Explore", Exploration(
        #     nan_report=True,
        #     plot_data=None,  # "All" | "weather" | "energy" | None | "seaborn"
        # )),
        # ("save_features", SaveORLoad(mode='save')),
        ("load_features", SaveORLoad(mode='load')),
        ("#4 Train Models", MultiRegXGBoostTraining( #ModelTraining
            target_column = 'renewable_generation_ratio',
            test_size=0.2,
            random_state=42,
            plot_results=True
        ))
    ])
    return pipe

if __name__ == "__main__":

    pipeline = build_pipeline(
        data_dir_or_paths='./data', #os.environ.get("DATA_DIR", "."),
        target_column='renewable_generation_ratio'
    )
    output = pipeline.fit_transform(None)
    pass # for breakpoint
    print("[main] Pipeline completed.")