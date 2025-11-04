import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray


# optimize for a single day - assumes aligned ratio and availability values
def optimize(
        car_name_list: List[str], 
        hrs_bounds: Tuple[int, int], 
        required_charging_hrs: int, 
        grid_cars_limit: int, 
        availability_dict: Dict[str, ArrayLike],
        renewability_ratio: ArrayLike,
        grb_verbose: bool = False
    ) -> Dict | None:
    
    renew_share = np.asarray(renewability_ratio, dtype=float).reshape(-1)
    
    # reassign variables
    V = car_name_list
    T = list(range(hrs_bounds[0], hrs_bounds[1]))
    required_hours = required_charging_hrs
    grid_limit = grid_cars_limit

    # --- Model ---
    m = gp.Model("renewable_charging")
    if not grb_verbose:
        m.setParam('OutputFlag', 0)

    # Decision variables
    X = m.addVars(V, T, vtype=GRB.BINARY, name="X")

    # Constraints
    # 1) Availability
    m.addConstrs((X[v, i] <= availability_dict[v, i] for v in V for i in T), name="availability")

    # 2) Full charge for each car
    m.addConstrs((gp.quicksum(X[v, i] for i in T) == required_hours for v in V), name="charge_time")

    # 3) Grid limit per hour
    m.addConstrs((gp.quicksum(X[v, i] for v in V) <= grid_limit for i in T), name="grid_limit")

    # Objective: maximize total renewable-weighted charging
    m.setObjective(gp.quicksum(renew_share[i] * X[v, i] for v in V for i in T), GRB.MAXIMIZE)

    # Solve
    m.optimize()

    # --- Results ---
    if m.status == GRB.OPTIMAL:
        result_dict = dict(charging_hours_per_vehicle=dict())
        for v in V:
            result_dict["charging_hours_per_vehicle"][v] = [i for i in T if X[v, i].x > 0.5]
        # Average renewable share
        result_dict['total_renew'] = sum(renew_share[i] * X[v, i].x for v in V for i in T)
        result_dict['total_charges'] = len(V) * required_hours
        result_dict['avg_renew_share'] = result_dict['total_renew'] / result_dict['total_charges']
        return result_dict
    else:
        print("Model not optimal (status:", m.status, ")")
        return None
    

# optimize for multiple days, alignes availability and ratios itself
def optimize_multi(
        car_name_list: List[str], 
        hrs_bounds: Tuple[int, int], 
        required_charging_hrs: int, 
        grid_cars_limit: int, 
        availability_dict: Dict[str, ArrayLike],
        renewability_ratio: ArrayLike,
        pred_hour: int = 8,
        grb_verbose: bool = False
    ) -> Dict | None:
    
    renew_share = np.asarray(renewability_ratio, dtype=float)
    assert renew_share.ndim == 2 and renew_share.shape[1] == 24
    
    # reassign variables
    V = car_name_list
    T = list(range(hrs_bounds[0], hrs_bounds[1]))
    required_hours = required_charging_hrs
    grid_limit = grid_cars_limit

    # alighn the ratoio to the availability ... r[0] -> r[pred_hour]
    renew_share_aligned = np.roll(renew_share, axis=1, shift=pred_hour)

    # global result
    result_dict = {
        'charging_hours_per_vehicle': [],
        'total_renew': [],
        'total_charges': [],
        'avg_renew_share': []
    }

    # compute optimization per day
    N_days = renew_share_aligned.shape[0]

    for day_idx in range(N_days):
        # --- Model ---
        m = gp.Model("renewable_charging")
        if not grb_verbose:
            m.setParam('OutputFlag', 0)

        # Decision variables
        X = m.addVars(V, T, vtype=GRB.BINARY, name="X")

        # Constraints
        # 1) Availability
        m.addConstrs((X[v, i] <= availability_dict[v, i] for v in V for i in T), name="availability")

        # 2) Full charge for each car
        m.addConstrs((gp.quicksum(X[v, i] for i in T) == required_hours for v in V), name="charge_time")

        # 3) Grid limit per hour
        m.addConstrs((gp.quicksum(X[v, i] for v in V) <= grid_limit for i in T), name="grid_limit")

        # Objective: maximize total renewable-weighted charging
        m.setObjective(gp.quicksum(renew_share_aligned[day_idx, i] * X[v, i] for v in V for i in T), GRB.MAXIMIZE)

        # Solve
        m.optimize()

        # --- Results ---
        if m.status == GRB.OPTIMAL:
            charging_hours_dict = {}
            for v in V:
                charging_hours_dict[v] = [i for i in T if X[v, i].x > 0.5]
            result_dict['charging_hours_per_vehicle'].append(charging_hours_dict)
            # Average renewable share
            result_dict['total_renew'].append(sum(renew_share_aligned[i] * X[v, i].x for v in V for i in T))
            result_dict['total_charges'].append(len(V) * required_hours)
            result_dict['avg_renew_share'].append(result_dict['total_renew'][-1] / result_dict['total_charges'][-1])
        else:
            print("Model not optimal (status:", m.status, ")")
            raise ValueError('Failed to find optimal solution')
    
    return result_dict


def demo_single_main() -> None:
    # --- Sets and parameters ---
    V = ["car1", "car2", "car3", "car4", "car5"]
    T = list(range(24))                   # hours 0–14
    required_hours = 4                    # each car must charge 4 hours
    grid_limit = 2                        # max 2 cars at once

    # binary availability matrix y[v,i]
    y = {(v, i): 1 for v in V for i in T}  # all available (you can modify)
    # Example: car1 not available at hours 0–2
    for i in range(3):
        y["car1", i] = 0

    # renewable share per hour (arbitrary example)
    renew_share = [0.1, 0.15, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.9, 0.85, 0.75, 0.5, 0.4, 0.2, 0.1, 0.1, 0.15, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.9, 0.85, 0.75, 0.5, 0.4, 0.2, 0.1]

    # run the optimization
    result = optimize(
        car_name_list=V,
        hrs_bounds=(0, 24),
        grid_cars_limit=grid_limit,
        required_charging_hrs=required_hours,
        availability_dict=y,
        renewability_ratio=renew_share
    )

    print(result)


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


def demo_multi_main() -> None:
    import pickle

    with open('./test_results.pkl', 'rb') as file:
        model_result = pickle.load(file)

    renewable_ratio_pred = model_result['preds']
    renewable_ratio_gt = model_result['gt']

    # --- Sets and parameters ---
    V = ["car1", "car2", "car3", "car4", "car5"]
    T = list(range(24))                   # hours 0–14
    required_hours = 4                    # each car must charge 4 hours
    grid_limit = 2                        # max 2 cars at once

    # binary availability matrix y[v,i]
    y = {(v, i): 1 for v in V for i in T}  # all available (you can modify)
    # Example: car1 not available at hours 0–2
    for i in range(3):
        y["car1", i] = 0

    # optimize based on prediction
    print('optimizing based on prediction ...')
    result_pred = optimize_multi(
        car_name_list=V,
        hrs_bounds=(0, 24),
        grid_cars_limit=grid_limit,
        required_charging_hrs=required_hours,
        availability_dict=y,
        renewability_ratio=renewable_ratio_pred,
        pred_hour=0
    )
    avg_green_pred = np.asarray(result_pred['avg_renew_share']).mean()
    _, avg_green_pred_real = score_charging_plan(result_pred['charging_hours_per_vehicle'], result_pred['total_charges'], renewable_ratio_gt, 0)
    print(f'   EXPECTED avg_renew_share: {avg_green_pred}')
    print(f'   REAL avg_renew_share: {avg_green_pred_real}')

    # optimize based on ground truth
    print('optimizing based on prediction ...')
    result_gt = optimize_multi(
        car_name_list=V,
        hrs_bounds=(0, 24),
        grid_cars_limit=grid_limit,
        required_charging_hrs=required_hours,
        availability_dict=y,
        renewability_ratio=renewable_ratio_gt,
        pred_hour=0
    )
    avg_green_gt = np.asarray(result_gt['avg_renew_share']).mean()
    print(f'   REAL avg_renew_share: {avg_green_gt}')


if __name__ == '__main__':
    #demo_single_main()
    demo_multi_main()
    print('done')
