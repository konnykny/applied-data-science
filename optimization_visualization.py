import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

def plot_optimization(
    ev_sessions,
    green_share,
    green_share_pred,
    working_hours=[*range(7, 12), *range(13, 17)],
    #max_evs=12,
    session_power = 11, 
    start_hour: int = 11,
    power_capacity=11*12,  # 12 is the number of cars
    save_fig=False,
    block=False,
    name='Optimized',
):
    fig, ax_power = plt.subplots(figsize=(16,8))
    plt.title(f"{name} Charging Schedule for 24h", fontsize=25)
    
    #ev_sessions = ev_sessions.sort_values(by='ev_id').reset_index(drop=True)

    hours = range(start_hour, start_hour + 24)
    evs = ev_sessions.ev_id.unique()
    num_evs = len(evs)
    
    # green share
    ax_green = ax_power.twinx()
    #ax_green.fill_between(hours, green_share, color="forestgreen", 
    #                      linewidth=3, alpha=.5, zorder=0)
    ax_green.plot(hours, green_share, color="red", zorder=12, linestyle='--',
                          linewidth=3, label="Carbon-free ratio GT [%]")
    ax_green.plot(hours, green_share_pred, color="forestgreen", zorder=12,
                          linewidth=3, label="Carbon-free ratio pred [%]")
    ax_green.set_ylabel("Carbon-free ratio [%]", fontsize=20)
    ax_green.set_ylim(max(0, min(green_share) * .8), max(85, max(green_share) * 1.1))
    
    # layers
    ax_green.set_zorder(2)
    ax_power.set_zorder(1)
    ax_power.patch.set_alpha(0.0)
    
    # working hours
    for hour in hours:
        if hour % 24 in working_hours:
            if hour % 24 > 12:
                ax_power.axvspan(hour, hour+1, facecolor="whitesmoke", edgecolor="none", alpha=.6, zorder=5)
            else:
                ax_power.axvspan(hour, hour+1, facecolor="silver", edgecolor="none", alpha=.6, zorder=5)

    # for i, hour in enumerate(working_hours):
    #     if i == 0:
    #         ax_power.axvspan(hour, hour+1, facecolor="silver", edgecolor="none", 
    #                          alpha=.6, zorder=5, label="Working hours")
    #     elif hour > 12:
    #         ax_power.axvspan(hour, hour+1, facecolor="whitesmoke", edgecolor="none", alpha=.6, zorder=5)
    #     else:
    #         ax_power.axvspan(hour, hour+1, facecolor="silver", edgecolor="none", alpha=.6, zorder=5)
    
    # EVs
    ev_colors = np.vstack([cm.get_cmap(n).colors for n in ["tab20", "tab20b", "tab20c"]])[:25]
    ev_colors = np.concatenate([ev_colors[::2], ev_colors[1::2]])
    bases = np.array([0 for _ in hours])
    for ev, row in ev_sessions.iterrows():
        shifted_sessions = [s + start_hour for s in row.sessions]
        base = [bases[i] for i in row.sessions]
        ax_power.bar(shifted_sessions, session_power, width=1, bottom=base, 
                     color=ev_colors[ev], edgecolor="gray", align='edge', alpha=.9, zorder=10)
        bases += [session_power if i in shifted_sessions else 0 for i in hours]

    # power
    ax_power.plot(hours, [power_capacity for _ in hours], 
                  linestyle="--", linewidth=2, zorder=12, label="Site limit [kW]")
    #ax_power.plot(range(24), bases, label="Aggregated charging power (kW)")
    ax_power.set_xlabel("Time [hrs]", fontsize=20)
    ax_power.set_ylabel("Power [kW]", fontsize=20)
    ax_power.set_xlim(hours[0], hours[-1])
    ax_power.set_ylim(0, power_capacity*1.3)
    ax_power.set_xticks(hours)
    ax_power.set_xticklabels([h % 24 for h in hours], fontsize=18)
    ax_power.set_yticks(range(0, power_capacity+1, session_power))
    ax_power.tick_params(axis='y', labelsize=18)
    
    # legend left
    h1, l1 = ax_power.get_legend_handles_labels()
    h2, l2 = ax_green.get_legend_handles_labels()
    legend = ax_power.legend(h1 + h2, l1 + l2, loc="upper left")
    ax_power.add_artist(legend)
    
    # legend EVs right
    ev_handles = [Rectangle((0, 0), 1, 1, facecolor=ev_colors[ev], edgecolor="none") for ev in range(num_evs)]
    ax_power.legend(ev_handles, evs, title="EVs", loc="upper right", ncol=-(num_evs//-3))
    
    plt.savefig(f"{name}_optimization_schedule.png")
    plt.show(block=block)

