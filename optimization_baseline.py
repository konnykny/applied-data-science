import matplotlib.pyplot as plt
import numpy as np
import pickle
from typing import Dict, List, Tuple

def baseline_scheduling(
    car_name_list: List[str],
    hrs_bounds: Tuple[int, int],
    required_charging_hrs: int,
    grid_cars_limit: int,
    availability_dict: Dict[str, List[int]],
    pred_hour: int = 8
) -> Dict:
    V = car_name_list
    T = list(range(hrs_bounds[0], hrs_bounds[1]))
    required_hours = required_charging_hrs
    grid_limit = grid_cars_limit
    charging_schedule = {v: [] for v in V}
    hourly_usage = {t: 0 for t in T}

    # Try to assign charging slots starting from pred_hour
    for v in V:
        # Get available hours for the car
        available_hours = [t for t in T if availability_dict[v, t] == 1]

        # Sort available hours based on circular distance from pred_hour
        available_hours_sorted = sorted(available_hours, key=lambda x: (x - pred_hour) % 24)

        assigned_hours = []
        for hour in available_hours_sorted:
            if hourly_usage[hour] < grid_limit:
                assigned_hours.append(hour)
                hourly_usage[hour] += 1
                if len(assigned_hours) == required_hours:
                    break
        if len(assigned_hours) < required_hours:
            raise ValueError(f"Cannot schedule {required_hours} hours for car {v} within the constraints.")

        # Store hours as offsets from pred_hour
        charging_schedule[v] = [(h - pred_hour) % 24 for h in assigned_hours]

    return charging_schedule

def score_charging_plan(charging_schedule_per_day: List[Dict], total_charges_per_day: List[int], renewable_ratio_per_day, pred_hour: int):
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

def visualize_schedule(charging_schedule: Dict, car_name_list: List[str], pred_hour: int = 8):
    print(charging_schedule)
    fig, ax = plt.subplots(figsize=(10, 5))

    for car, hours in charging_schedule.items():
        for hour in hours:
            # Determine the position of the hour on the circular axis
            pos = hour if hour >= pred_hour else hour + 24
            ax.broken_barh([(pos, 1)], (car_name_list.index(car) - 0.4, 0.8), facecolors='tab:blue')

    ax.set_yticks(range(len(car_name_list)))
    ax.set_yticklabels(car_name_list)
    ax.set_xlim(pred_hour, pred_hour + 24)
    ax.set_xticks(range(pred_hour, pred_hour + 24))
    ax.set_xticklabels([f"{h % 24}:00" for h in range(pred_hour, pred_hour + 24)])
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Car')
    ax.set_title('Charging Schedule Visualization')

    plt.grid(True)
    plt.show(block=False)

def demo_baseline_main():
    # --- Sets and parameters ---
    V = ["car1", "car2", "car3", "car4", "car5"]
    T = list(range(24))                   # hours 0-23
    required_hours = 4                    # each car must charge 4 hours
    grid_limit = 2                        # max 2 cars at once
    pred_hour = 8                         # prediction hour
    # binary availability matrix y[v,i]
    y = {(v, i): 1 for v in V for i in T}  # all available (you can modify)
    # Example: car1 not available at hours 0-2
    for i in range(3):
        y["car1", i] = 0

    # Load the prediction and ground truth data
    with open('./test_results.pkl', 'rb') as file:
        model_result = pickle.load(file)

    renewable_ratio_pred = model_result['preds']
    renewable_ratio_gt = model_result['gt']

    # Generate charging schedules for each day
    charging_schedules = []
    total_charges_per_day = []
    for _ in range(renewable_ratio_pred.shape[0]):
        try:
            charging_schedule = baseline_scheduling(
                car_name_list=V,
                hrs_bounds=(0, 24),
                required_charging_hrs=required_hours,
                grid_cars_limit=grid_limit,
                availability_dict=y,
                pred_hour=pred_hour
            )
            charging_schedules.append(charging_schedule)
            total_charges_per_day.append(len(V) * required_hours)
        except ValueError as e:
            print(e)
            return

    # Score the charging schedules using predicted and actual renewable ratios
    daily_renew_ratios_pred, avg_renew_ratio_pred = score_charging_plan(charging_schedules, total_charges_per_day, renewable_ratio_pred, pred_hour)
    daily_renew_ratios_gt, avg_renew_ratio_gt = score_charging_plan(charging_schedules, total_charges_per_day, renewable_ratio_gt, pred_hour)

    print("Average Renewable Ratio (Predicted):", avg_renew_ratio_pred)
    print("Average Renewable Ratio (Actual):", avg_renew_ratio_gt)

    # Visualize the charging schedule for the first day
    visualize_schedule(charging_schedules[0], V, pred_hour)

if __name__ == '__main__':
    demo_baseline_main()
