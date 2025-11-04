import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy.typing import ArrayLike, NDArray
from typing import Dict, List, Tuple

from optimization import optimize
from optim_scenarios import example01
from optimization_baseline import baseline_scheduling, visualize_schedule


def score_charging_plan(charging_schedule_per_day: List[Dict], total_charges_per_day: List[int], renewable_ratio_per_day: ArrayLike, pred_hour: int) -> Tuple[NDArray, float]:

    renewable_ratio_per_day = np.asarray(renewable_ratio_per_day, dtype=float)
    renewable_ratio_per_day_aligned = np.roll(renewable_ratio_per_day, axis=1, shift=pred_hour)

    daily_renew_ratios = np.zeros((renewable_ratio_per_day_aligned.shape[0],), dtype=float)
    for day_idx in range(len(daily_renew_ratios)):
        schedule_per_car = charging_schedule_per_day[day_idx]
        total_charges = total_charges_per_day[day_idx]
        renewable_ratio = renewable_ratio_per_day_aligned[day_idx, :]

        for v in schedule_per_car.keys():
            for h in schedule_per_car[v]:
                daily_renew_ratios[day_idx] += renewable_ratio[h]
        daily_renew_ratios[day_idx] /= total_charges

    return daily_renew_ratios, daily_renew_ratios.mean()


def test_main() -> None:
    import pickle

    with open('./test_results.pkl', 'rb') as file:
        model_result = pickle.load(file)

    renewable_ratio_pred = model_result['preds']
    renewable_ratio_gt = model_result['gt']

    print(renewable_ratio_gt.shape, renewable_ratio_pred.shape)

    # --- Sets and parameters ---
    V = example01.config['car_names_list']
    T = example01.config['T']
    required_hours = example01.config['required_charging_hours']
    grid_limit = example01.config['grid_cars_limit']

    # binary availability matrix y[v,i]
    y = {(v, i): example01.config['availability_dict'][v][i] for v in V for i in T} 

    # optimize based on prediction
    print('optimizing based on prediction ...')
    result_pred = optimize(
        car_name_list=V,
        hrs_bounds=(0, 24),
        grid_cars_limit=grid_limit,
        required_charging_hrs=required_hours,
        availability_dict=y,
        renewability_ratio=renewable_ratio_pred,
        pred_hour=8
    )
    avg_green_pred = np.asarray(result_pred['avg_renew_share']).mean()
    _, avg_green_pred_real = score_charging_plan(result_pred['charging_hours_per_vehicle'], result_pred['total_charges'], renewable_ratio_gt, 0)
    print(f'   EXPECTED avg_renew_share: {avg_green_pred}')
    print(f'   REAL avg_renew_share: {avg_green_pred_real}')

    # optimize based on ground truth
    print('optimizing based on ground truth ...')
    result_gt = optimize(
        car_name_list=V,
        hrs_bounds=(0, 24),
        grid_cars_limit=grid_limit,
        required_charging_hrs=required_hours,
        availability_dict=y,
        renewability_ratio=renewable_ratio_gt,
        pred_hour=8
    )
    avg_green_gt = np.asarray(result_gt['avg_renew_share']).mean()
    print(f'   REAL avg_renew_share: {avg_green_gt}')

    # get the baseline charging schedule
    print(f'computing baseline schedule ...')
    schedule_baseline_one_day = baseline_scheduling(
        car_name_list=V,
        hrs_bounds=(0, 24),
        grid_cars_limit=grid_limit,
        availability_dict=y,
        required_charging_hrs=required_hours
    )
    visualize_schedule(schedule_baseline_one_day, car_name_list=V, pred_hour=8)
    N_days = 1 if renewable_ratio_gt.ndim == 1 else renewable_ratio_gt.shape[0]
    schedule_baseline = {
        'charging_hours_per_vehicle': [schedule_baseline_one_day for _ in range(N_days)],
        'total_charges': [len(V) * required_hours for _ in range(N_days)]
    }
    _, avg_green_baseline = score_charging_plan(schedule_baseline['charging_hours_per_vehicle'], schedule_baseline['total_charges'], renewable_ratio_gt, 0)
    print(f'   REAL avg_renew_share: {avg_green_baseline}')

if __name__ == '__main__':
    test_main()
    print('done')

