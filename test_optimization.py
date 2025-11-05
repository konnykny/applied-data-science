import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy.typing import ArrayLike, NDArray
from typing import Dict, List, Tuple

from optimization import optimize
from optim_scenarios import example01
from optimization_baseline import baseline_scheduling, visualize_schedule


def score_charging_plan(charging_schedule_per_day: List[Dict], total_charges_per_day: List[int], renewable_ratio_per_day: ArrayLike) -> Tuple[NDArray, float]:

    renewable_ratio_per_day = np.asarray(renewable_ratio_per_day, dtype=float)

    daily_renew_ratios = np.zeros((renewable_ratio_per_day.shape[0],), dtype=float)
    for day_idx in range(len(daily_renew_ratios)):
        schedule_per_car = charging_schedule_per_day[day_idx]
        total_charges = total_charges_per_day[day_idx]
        renewable_ratio = renewable_ratio_per_day[day_idx, :]

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
    green_pred_scores, avg_green_pred_real = score_charging_plan(result_pred['charging_hours_per_vehicle'], result_pred['total_charges'], renewable_ratio_gt)
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
    #green_gt_scores, avg_green_gt = score_charging_plan(result_gt['charging_hours_per_vehicle'], result_gt['total_charges'], renewable_ratio_gt)
    
    green_gt_scores = np.asarray(result_gt['avg_renew_share'])
    avg_green_gt = green_gt_scores.mean()
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
    green_baseline_scores, avg_green_baseline = score_charging_plan(schedule_baseline['charging_hours_per_vehicle'], schedule_baseline['total_charges'], renewable_ratio_gt)
    print(f'   REAL avg_renew_share: {avg_green_baseline}')

    # visualizations
    fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    ax[0].plot(renewable_ratio_pred.mean(axis=1) * 100, c='blue', linestyle='-', label='Mean of daily predictions [%]')
    ax[0].plot(renewable_ratio_gt.values.mean(axis=1) * 100, c='red', linestyle='--', label='Mean of daily GT [%]')

    differences = renewable_ratio_gt.values.mean(axis=1) - renewable_ratio_pred.mean(axis=1)

    red_indices = [i for i, diff in enumerate(differences) if diff < 0]
    blue_indices = [i for i, diff in enumerate(differences) if diff >= 0]

    ax[1].bar(red_indices, [100*differences[i] for i in red_indices], color='red', label='Pred > GT')

    ax[1].bar(blue_indices, [100*differences[i] for i in blue_indices], color='blue', label='Pred < GT')

    score_pred_diffs = green_pred_scores - green_baseline_scores
    score_gt_diffs = green_gt_scores - green_baseline_scores

    green_indices = [i for i, diff in enumerate(score_pred_diffs) if diff >= 0]
    red_indices = [i for i, diff in enumerate(score_pred_diffs) if diff < 0]

    ax[2].bar(green_indices, [100*score_pred_diffs[i] for i in green_indices], color='green', label='Improvement against baseline')
    ax[2].bar(red_indices, [100*score_pred_diffs[i] for i in red_indices], color='red', label='Not an improvement against baseline')
    ax[2].plot(range(len(score_gt_diffs)), 100*score_gt_diffs, c='blue', linestyle='-', alpha=0.5, label='Maximum achievable improvement (with perfect model / GT)')

    ax[1].axhline(0, color='black', linewidth=0.8)
    ax[2].axhline(0, color='black', linewidth=0.8)

    ax[0].set_title('Predicted x GT green ratio')
    ax[1].set_title('Predicted x GT green ratio error')
    ax[2].set_title('Optimized x Baseline schedule green ratio comparison')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    ax[0].set_ylabel('%')
    ax[1].set_ylabel('%')
    ax[2].set_ylabel('%')
    
    ax[0].set_xlabel('days')
    ax[1].set_xlabel('days')
    ax[2].set_xlabel('days')

    plt.show()

if __name__ == '__main__':
    test_main()
    print('done')

