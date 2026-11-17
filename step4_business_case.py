# -*- coding: utf-8 -*-
"""
BESS case study - Step 4: business case (NPV, IRR, payback)
300 MW battery at Hollandse Kust Noord landstation, 380 kV

Compares four cases: 2h vs 4h battery, each with a FIRM or a TDTR grid contract.

The year-1 revenues come from step 3 output (base case: 50 MW aFRR,
25 MW FCR, rest arbitrage). This is how real models work: a technical
dispatch model feeds a separate financial model.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq    # finds where a function crosses zero (used for IRR)

POWER_MW = 300
LIFETIME = 15             # years of operation
WACC = 0.08               # discount rate: the return investors require (pre-tax, nominal)

# --- 1. Year-1 revenues from step 3, EUR million (base case row) -----------
REVENUE = {
    "2h": {"arbitrage": 18.76, "afrr": 11.23, "fcr": 3.12},
    "4h": {"arbitrage": 29.47, "afrr": 11.23, "fcr": 3.12},
}

# How revenues change each year
ARBITRAGE_DECLINE = 0.02   # spreads narrow as more batteries are built (-2%/yr)
BALANCING_DECLINE = 0.10   # aFRR/FCR prices keep falling (-10%/yr); they fell ~4x since 2022
CAPACITY_FADE = 0.02       # battery loses ~2% of its MWh per year (hits arbitrage only)
TRADING_FEE = 0.07         # optimiser / route-to-market fee: share of revenue

# Intraday + imbalance trading on top of day-ahead, EUR million per MW per year.
# NOT modelled by me, so it starts at 0. KYOS (Feb 2025) estimates 2026 NL revenue
# for a 2h battery at ~105k/MW day-ahead vs ~308k/MW intraday + imbalance (no grid
# fees, no degradation). Use this to test scenarios, e.g. 0.05 = EUR 50k/MW/yr.
INTRADAY_UPLIFT = 0.1446

# --- 2. Capital cost (capex), EUR million ----------------------------------
# Battery system incl. PCS, MV transformers, civil works (turnkey, EUR per kWh).
# Based on Modo Energy's 2026 global survey (median 2h ~$325/kWh, 4h ~$201/kWh),
# rounded down for falling prices. Check against recent European project news.
CAPEX_PER_KWH = {"2h": 280, "4h": 180}
GRID_CONNECTION_CAPEX = 35   # 380 kV bay, 380/33 kV transformers, HV works (assumption)
OPEX_SHARE = 0.015           # O&M, insurance, land lease: 1.5% of battery capex per year

# --- 3. TenneT grid tariffs (charged on grid offtake capacity) -------------
# Modo Energy: ~EUR 116k per MW per year at extra-high voltage (380 kV) with a
# firm contract. TDTR (85% firm, TenneT may curtail 15% of the time) gives a
# 43% discount on contracted capacity, up to 62% with smart charging timing.
GRID_TARIFF_FIRM = 0.116                      # EUR million per MW per year
CONTRACTS = {"Firm": 0.0, "TDTR": 0.43}       # discount on the grid tariff
TDTR_REVENUE_LOSS = 0.05                      # arbitrage lost to curtailment (assumption)


def cash_flows(duration, contract):
    """Return a list of yearly cash flows, EUR million. Year 0 = construction."""
    energy_mwh = POWER_MW * int(duration[0])
    battery_capex = energy_mwh * 1000 * CAPEX_PER_KWH[duration] / 1e6
    total_capex = battery_capex + GRID_CONNECTION_CAPEX

    discount = CONTRACTS[contract]
    curtail_loss = TDTR_REVENUE_LOSS if contract == "TDTR" else 0.0
    rev = REVENUE[duration]

    flows = [-total_capex]                     # year 0: we pay for the plant
    for year in range(1, LIFETIME + 1):
        n = year - 1                           # years since start
        arb_mw = POWER_MW - 75                 # MW left for trading (300 - 50 aFRR - 25 FCR)
        arbitrage = ((rev["arbitrage"] + INTRADAY_UPLIFT * arb_mw) * (1 - curtail_loss)
                     * (1 - ARBITRAGE_DECLINE) ** n * (1 - CAPACITY_FADE) ** n)
        balancing = (rev["afrr"] + rev["fcr"]) * (1 - BALANCING_DECLINE) ** n
        revenue = (arbitrage + balancing) * (1 - TRADING_FEE)

        grid_cost = GRID_TARIFF_FIRM * POWER_MW * (1 - discount)
        opex = OPEX_SHARE * battery_capex
        flows.append(revenue - grid_cost - opex)
    return flows, total_capex


def npv(rate, flows):
    """Net present value: each year's cash divided by (1+rate)^year, then summed."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(flows))


def irr(flows):
    """Internal rate of return: the discount rate at which NPV = 0."""
    try:
        return brentq(lambda r: npv(r, flows), -0.99, 1.0)
    except ValueError:                         # no solution: project never pays back
        return np.nan


def payback(flows):
    """First year in which cumulative cash turns positive."""
    cumulative = np.cumsum(flows)
    positive = np.where(cumulative > 0)[0]
    return int(positive[0]) if len(positive) else None


# --- 4. Run all four cases --------------------------------------------------
rows, all_flows = [], {}
for duration in ["2h", "4h"]:
    for contract in CONTRACTS:
        flows, capex = cash_flows(duration, contract)
        name = f"{duration} / {contract}"
        all_flows[name] = flows
        rows.append({
            "Case": name,
            "Capex (EUR M)": capex,
            "Year-1 cash flow (EUR M)": flows[1],
            "NPV @8% (EUR M)": npv(WACC, flows),
            "IRR": irr(flows),
            "Payback (yr)": payback(flows),
        })

results = pd.DataFrame(rows)
pd.set_option("display.width", 140)
print(results.round(2).to_string(index=False))

# --- 5. Break-even question -----------------------------------------------
# The model only includes day-ahead trading and balancing CAPACITY payments.
# Real Dutch batteries also earn from intraday trading, imbalance (passive
# balancing) and aFRR activation energy. How much extra would be needed?
annuity = sum(1 / (1 + WACC) ** t for t in range(1, LIFETIME + 1))
print("\nExtra yearly revenue needed for NPV = 0 (EUR k per MW per year):")
for name, flows in all_flows.items():
    gap = -npv(WACC, flows) / annuity / POWER_MW * 1000
    print(f"  {name}: {max(gap, 0):.0f}")

# --- 6. Chart: cumulative cash over the project life -----------------------
plt.figure(figsize=(10, 5))
for name, flows in all_flows.items():
    plt.plot(range(LIFETIME + 1), np.cumsum(flows), marker="o", label=name)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Cumulative cash flow, 300 MW BESS (undiscounted)")
plt.xlabel("Year")
plt.ylabel("EUR million")
plt.legend()
plt.grid(alpha=0.3)
plt.show()