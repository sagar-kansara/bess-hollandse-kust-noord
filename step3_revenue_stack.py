"""
BESS case study - Step 3: revenue stack = arbitrage + aFRR + FCR

Idea: the battery splits its 300 MW.
  - Some MW are RESERVED for balancing (aFRR, FCR). TenneT pays a capacity
    price for every hour that MW is on standby, whether it is used or not.
  - The remaining MW trade on the day-ahead market (the step 2 model).

Simplification (state it in your report): reserved MW are taken out of
arbitrage completely, together with their share of energy. Real operators
co-optimise and squeeze more out of the same battery, so this is conservative.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# --- 1. Assumptions --
POWER_MW = 300
DURATIONS = [2, 4]                 # hours: 600 MWh and 1200 MWh
RTE = 0.88
ETA = np.sqrt(RTE)
DEGRADATION_COST = 10              # EUR/MWh discharged

# Balancing prices, EUR per MW per hour of availability.
# aFRR: Modo Energy reports ~27 EUR/MW/h for 2026 (down from 105 in 2022).
# FCR: European auction, roughly 15 EUR/MW/h recently. Both are falling as
# more batteries enter: check them against TenneT / regelleistung.net data.
AFRR_PRICE = 27
FCR_PRICE = 15
AVAILABILITY = 0.95                # share of hours the battery is actually available

FCR_MW = 25                        # small slice: the Dutch FCR need is only ~100 MW in total
AFRR_OPTIONS = [0, 50, 100, 150]   # how many MW we offer to aFRR (the scenario we vary)

HOURS_PER_YEAR = 8760

# --- 2. Load 2025 prices ---------------------------------------------------
df = pd.read_csv("Netherlands.csv", parse_dates=["Datetime (Local)"])
df = df[["Datetime (Local)", "Price (EUR/MWhe)"]]
df.columns = ["time", "price"]
prices = df.set_index("time").loc["2025"]["price"]


def optimise_day(day_prices, power, energy):
    """Same LP as step 2, with power as an input. Returns market profit in EUR."""
    n = len(day_prices)
    p = day_prices.values
    cost = np.concatenate([p, -p + DEGRADATION_COST, np.zeros(n)])
    A_eq = np.zeros((n + 1, 3 * n))
    b_eq = np.zeros(n + 1)
    for t in range(n):
        A_eq[t, t] = -ETA
        A_eq[t, n + t] = 1 / ETA
        A_eq[t, 2 * n + t] = 1
        if t == 0:
            b_eq[t] = 0.5 * energy
        else:
            A_eq[t, 2 * n + t - 1] = -1
    A_eq[n, 3 * n - 1] = 1
    b_eq[n] = 0.5 * energy
    bounds = [(0, power)] * (2 * n) + [(0, energy)] * n
    result = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    discharge = result.x[n:2 * n]
    return -result.fun + DEGRADATION_COST * discharge.sum()


# --- 3. Arbitrage revenue PER MW, once for each duration -------------------
# Step 2 showed revenue scales linearly with size, so we optimise a 1 MW
# battery and multiply. This makes the script much faster.
arb_per_mw = {}
for d in DURATIONS:
    arb_per_mw[d] = sum(optimise_day(dp, 1, d)
                        for _, dp in prices.groupby(prices.index.date))
    print(f"Arbitrage, {d}h battery: EUR {arb_per_mw[d] / 1000:.0f}k per MW per year")

# --- 4. Build the revenue stack for every scenario -------------------------
rows = []
for d in DURATIONS:
    for afrr_mw in AFRR_OPTIONS:
        arb_mw = POWER_MW - afrr_mw - FCR_MW            # MW left for trading
        rows.append({
            "Battery": f"{POWER_MW} MW / {POWER_MW * d} MWh",
            "aFRR MW": afrr_mw,
            "Arbitrage": arb_mw * arb_per_mw[d] / 1e6,
            "aFRR": afrr_mw * AFRR_PRICE * HOURS_PER_YEAR * AVAILABILITY / 1e6,
            "FCR": FCR_MW * FCR_PRICE * HOURS_PER_YEAR * AVAILABILITY / 1e6,
        })

stack = pd.DataFrame(rows)
stack["Total"] = stack[["Arbitrage", "aFRR", "FCR"]].sum(axis=1)
stack["Balancing share"] = (stack["aFRR"] + stack["FCR"]) / stack["Total"]

pd.set_option("display.width", 120)
print("\nRevenue stack, EUR million per year:")
print(stack.round(2).to_string(index=False))

# --- 5. Stacked bar chart --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (name, group) in zip(axes, stack.groupby("Battery", sort=False)):
    group.set_index("aFRR MW")[["Arbitrage", "aFRR", "FCR"]].plot(
        kind="bar", stacked=True, ax=ax, rot=0)
    ax.set_title(name)
    ax.set_xlabel("MW offered to aFRR")
    ax.set_ylabel("EUR million per year")
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("Revenue stack by aFRR allocation (2025 prices, FCR fixed at 25 MW)")
plt.tight_layout()
plt.show()