import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
energy = pd.read_csv("data/spanish-cities-energy-consumption/energy_dataset.csv")
weather = pd.read_csv("data/spanish-cities-energy-consumption/weather_features.csv")

# Convert time columns to datetime
energy["time"] = pd.to_datetime(energy["time"])
weather["dt_iso"] = pd.to_datetime(weather["dt_iso"])

# Merge datasets (optional, depends if you want weather alongside energy)
df = energy.merge(weather, left_on="time", right_on="dt_iso", how="left")

# --- First look ---
print(energy.head())
print(weather.head())

# --- Example 1: Total load (forecast vs. actual) ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=energy["time"], y=energy["total load actual"],
                         mode="lines", name="Load Actual"))
fig.add_trace(go.Scatter(x=energy["time"], y=energy["total load forecast"],
                         mode="lines", name="Load Forecast"))
fig.update_layout(title="Total Load (Actual vs Forecast)",
                  xaxis_title="Time", yaxis_title="MW",
                  hovermode="x unified")
fig.show()

# --- Example 2: Price (day-ahead vs actual) ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=energy["time"], y=energy["price actual"],
                         mode="lines", name="Price Actual"))
fig.add_trace(go.Scatter(x=energy["time"], y=energy["price day ahead"],
                         mode="lines", name="Price Day Ahead"))
fig.update_layout(title="Price (Actual vs Day-Ahead)",
                  xaxis_title="Time", yaxis_title="€/MWh",
                  hovermode="x unified")
fig.show()

# --- Example 3: Energy generation breakdown ---
gen_cols = [col for col in energy.columns if col.startswith("generation ")]
df_gen = energy[["time"] + gen_cols].set_index("time")

fig = px.area(df_gen, x=df_gen.index, y=df_gen.columns,
              title="Energy Generation by Type",
              labels={"value": "MW", "variable": "Generation Type"})
fig.update_layout(hovermode="x unified")
fig.show()

# --- Example 4: Weather relation (temp & wind) ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["time"], y=df["temp"],
                         mode="lines", name="Temperature (°F)", yaxis="y1"))
fig.add_trace(go.Scatter(x=df["time"], y=df["wind_speed"],
                         mode="lines", name="Wind Speed (m/s)", yaxis="y2"))

fig.update_layout(
    title="Weather Features (Temp & Wind)",
    xaxis=dict(domain=[0.1, 0.9]),
    yaxis=dict(title="Temperature (°C)", side="left"),
    yaxis2=dict(title="Wind Speed (m/s)", overlaying="y", side="right"),
    hovermode="x unified"
)
fig.show()
