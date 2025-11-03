import gurobipy as gp
from gurobipy import GRB

# --- Sets and parameters ---
V = ["car1", "car2", "car3", "car4", "car5"]
T = list(range(15))                   # hours 0–14
required_hours = 4                    # each car must charge 4 hours
grid_limit = 2                        # max 2 cars at once

# binary availability matrix y[v,i]
y = {(v, i): 1 for v in V for i in T}  # all available (you can modify)
# Example: car1 not available at hours 0–2
for i in range(3):
    y["car1", i] = 0

# renewable share per hour (arbitrary example)
renew_share = {i: [0.1, 0.15, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.9, 0.85, 0.75, 0.5, 0.4, 0.2, 0.1][i] for i in T}

# --- Model ---
m = gp.Model("renewable_charging")

# Decision variables
X = m.addVars(V, T, vtype=GRB.BINARY, name="X")

# Constraints
# 1) Availability
m.addConstrs((X[v, i] <= y[v, i] for v in V for i in T), name="availability")

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
    print("\nOptimal schedule:")
    for v in V:
        hours = [i for i in T if X[v, i].x > 0.5]
        print(f"{v}: charge in hours {hours}")
    # Average renewable share
    total_renew = sum(renew_share[i] * X[v, i].x for v in V for i in T)
    total_charges = len(V) * required_hours
    avg_renew_share = total_renew / total_charges
    print(f"\nAverage renewable share of charging: {avg_renew_share}")
else:
    print("Model not optimal (status:", m.status, ")")
