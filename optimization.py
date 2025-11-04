import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import ArrayLike


# optimize for multiple days, alignes availability and ratios itself
def optimize(
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
    if renew_share.ndim == 1:
        renew_share = renew_share[None, :]  # create as 1xH -. only one day
    assert renew_share.ndim == 2 and renew_share.shape[1] == 24
    
    # reassign variables
    V = car_name_list
    T = list(range(hrs_bounds[0], hrs_bounds[1]))
    required_hours = required_charging_hrs
    grid_limit = grid_cars_limit

    # alighn the availability to the ratio ... a[pred_hour] -> a'[0]
    aligned_availability_dict = {}
    for v in V:
        for h in T:
            new_h = (h + pred_hour) % 24
            aligned_availability_dict[v, h] = availability_dict[v, new_h]

    # global result
    result_dict = {
        'charging_hours_per_vehicle': [],
        'total_renew': [],
        'total_charges': [],
        'avg_renew_share': []
    }

    # compute optimization per day
    N_days = renew_share.shape[0]

    for day_idx in range(N_days):
        # --- Model ---
        m = gp.Model("renewable_charging")
        if not grb_verbose:
            m.setParam('OutputFlag', 0)

        # Decision variables
        X = m.addVars(V, T, vtype=GRB.BINARY, name="X")

        # Constraints
        # 1) Availability
        m.addConstrs((X[v, i] <= aligned_availability_dict[v, i] for v in V for i in T), name="availability")

        # 2) Full charge for each car
        m.addConstrs((gp.quicksum(X[v, i] for i in T) == required_hours for v in V), name="charge_time")

        # 3) Grid limit per hour
        m.addConstrs((gp.quicksum(X[v, i] for v in V) <= grid_limit for i in T), name="grid_limit")

        # Objective: maximize total renewable-weighted charging
        m.setObjective(gp.quicksum(renew_share[day_idx, i] * X[v, i] for v in V for i in T), GRB.MAXIMIZE)

        # Solve
        m.optimize()

        # --- Results ---
        if m.status == GRB.OPTIMAL:
            charging_hours_dict = {}
            for v in V:
                charging_hours_dict[v] = [i for i in T if X[v, i].x > 0.5]
            result_dict['charging_hours_per_vehicle'].append(charging_hours_dict)
            # Average renewable share
            result_dict['total_renew'].append(sum(renew_share[i] * X[v, i].x for v in V for i in T))
            result_dict['total_charges'].append(len(V) * required_hours)
            result_dict['avg_renew_share'].append(result_dict['total_renew'][-1] / result_dict['total_charges'][-1])
        else:
            print("Model not optimal (status:", m.status, ")")
            raise ValueError('Failed to find optimal solution')
    
    return result_dict
