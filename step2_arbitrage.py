# -*- coding: utf-8 -*-
"""
BESS case study - Step 2: day-ahead arbitrage optimisation (2025)
 
Method: for each day, a linear program (LP) chooses how much to charge and
discharge in each hour to maximise profit, given that day's 24 prices.
Day-ahead prices are published the day before delivery, so knowing all 24
prices in advance is realistic for this market.
 
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog   # the LP solver, included with Anaconda
 
# --- 1. Load 2025 prices --------------------------------
df = pd.read_csv("Netherlands.csv", parse_dates=["Datetime (Local)"])
df = df[["Datetime (Local)", "Price (EUR/MWhe)"]]
df.columns = ["time", "price"]
prices = df.set_index("time").loc["2025"]["price"]
 
# --- 2. Battery assumptions ----------------
POWER_MW = 300            # max charge / discharge power
RTE = 0.88               # round-trip efficiency (typical LFP system, AC-AC)
ETA = np.sqrt(RTE)       # split equally between charging and discharging
DEGRADATION_COST = 10    # EUR per MWh discharged: a simple proxy for battery wear
START_SOC = 0.5          # battery starts and ends each day half full
 
 
def optimise_day(day_prices, energy_mwh):
    """Best charge/discharge schedule for one day. Returns profit (EUR) and MWh discharged."""
    n = len(day_prices)          # number of hours (24; 23 or 25 on clock-change days)
    p = day_prices.values
 
    # Decision variables, stacked into one long list x:
    #   x[0:n]     = charge power in each hour (MW)
    #   x[n:2n]    = discharge power in each hour (MW)
    #   x[2n:3n]   = state of charge at the end of each hour (MWh)
    # linprog MINIMISES, so we minimise cost = what we pay - what we earn
    cost = np.concatenate([p, -p + DEGRADATION_COST, np.zeros(n)])
 
    # Energy balance each hour:
    #   soc[t] = soc[t-1] + ETA*charge[t] - discharge[t]/ETA
    # rewritten as: soc[t] - soc[t-1] - ETA*charge[t] + discharge[t]/ETA = 0
    A_eq = np.zeros((n + 1, 3 * n))
    b_eq = np.zeros(n + 1)
    for t in range(n):
        A_eq[t, t] = -ETA             # charge
        A_eq[t, n + t] = 1 / ETA      # discharge
        A_eq[t, 2 * n + t] = 1        # soc[t]
        if t == 0:
            b_eq[t] = START_SOC * energy_mwh   # soc before the first hour
        else:
            A_eq[t, 2 * n + t - 1] = -1        # soc[t-1]
    # Last row: end the day at the same state of charge that was started with
    A_eq[n, 3 * n - 1] = 1
    b_eq[n] = START_SOC * energy_mwh
 
    # Limits on each variable: power between 0 and 50 MW, soc between 0 and full
    bounds = [(0, POWER_MW)] * (2 * n) + [(0, energy_mwh)] * n
 
    result = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    discharge = result.x[n:2 * n]
    profit = -result.fun + DEGRADATION_COST * discharge.sum()   # market profit, before wear cost
    return profit, discharge.sum()
 
 
# --- 3. Run for every day of 2025, for two battery sizes -----------------
results = {}
for energy in [600, 1200]:                       # 2-hour vs 4-hour battery
    daily_profit = []
    total_discharged = 0
    for day, day_prices in prices.groupby(prices.index.date):
        profit, discharged = optimise_day(day_prices, energy)
        daily_profit.append(profit)
        total_discharged += discharged
 
    annual = sum(daily_profit)
    cycles = total_discharged / energy           # full equivalent cycles per year
    results[energy] = pd.Series(daily_profit, index=pd.to_datetime(sorted(set(prices.index.date))))
 
    print(f"\n {POWER_MW} MW / {energy} MWh ({energy // POWER_MW}h):")
    print(f"  Annual arbitrage revenue: EUR {annual / 1e6:.2f} million")
    print(f"  Per MW of power:          EUR {annual / POWER_MW / 1000:.0f}k per MW per year")
    print(f"  Full cycles per year:     {cycles:.0f}  ({cycles / 365:.2f} per day)")
 
# --- 4. Plot monthly revenue for both sizes ------------------------------
monthly = pd.DataFrame({f"{e} MWh": results[e].resample("MS").sum() / 1e6 for e in results})
monthly.index = monthly.index.strftime("%b")
monthly.plot(kind="bar", figsize=(10, 4))
plt.title("Day-ahead arbitrage revenue by month, 50 MW battery, 2025")
plt.ylabel("EUR million")
plt.grid(axis="y", alpha=0.3)
plt.show()