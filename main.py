#%%
import os
from sklearn.pipeline import Pipeline

from preprocessing import Preprocessing
from exploration import Exploration
from features import FeatureExtraction, SaveORLoad
from modeling import ModelTraining


def build_pipeline(data_dir_or_paths, ffill_amt=3, sampling_interval=1, drop_na=False,
                   load_pickle_instead=None, save_pickle=True,
                   N_past_values=72, nan_report=True, plot_data="All",
                   target_column=None, test_size=0.2, random_state=42, plot_results=True, drop_energy_details=True):
    pipe = Pipeline(steps=[
        ("#1 Preprocessing", Preprocessing(
            data_path=data_dir_or_paths,
            ffill_amt=ffill_amt,
            sampling_interval=sampling_interval,
            drop_na=drop_na,
            load_pickle_instead=load_pickle_instead,
            save_pickle=save_pickle,
            drop_energy_details=drop_energy_details,
        )),
        # ("#2 Explore", Exploration(
        #     nan_report=nan_report,
        #     plot_data=plot_data
        # )),
        ("#3 Calculate features", FeatureExtraction(
            target_column, 
            N_past_values=N_past_values, 
            N_future_values=24, 
            weather_columns=['temp', 'temp_max'], 
        )),
        ("save_features", SaveORLoad(mode='save')),
        #("save_features", SaveORLoad(mode='load')),
        ("#4 Train Models", ModelTraining(
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            plot_results=plot_results
        ))
    ])
    return pipe


if __name__ == "__main__":
    data_dir = './data'#os.environ.get("DATA_DIR", ".")

    target_col = 'renewable_generation_ratio'

    pipeline = build_pipeline(
        data_dir_or_paths=data_dir,
        ffill_amt=3,
        drop_na=False, #drop rows with any NaN after ffill
        load_pickle_instead=f'{data_dir}\preprocessed_dataframe.pkl',
        save_pickle=True,
        N_past_values=[24, 24*7],
        nan_report=True,
        plot_data="energy", # "All" | "weather" | "energy" | None | "seaborn"
        target_column=target_col,
        test_size=0.2,
        random_state=42,
        plot_results=True,
        drop_energy_details=False
    )

    output = pipeline.fit_transform(None)
    pass
    print("[main] Pipeline completed.")
