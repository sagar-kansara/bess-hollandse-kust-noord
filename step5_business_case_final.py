# -*- coding: utf-8 -*-
"""
BESS case study - Step 5: final business case with the engineered equipment
300 MW BESS at Hollandse Kust Noord, 380 kV, TDTR contract

What changed since step 4:
  1. Capex is built from the EQUIPMENT in the engineering workbook (installed
     containers, PCS/MV skids, HV connection) instead of one turnkey EUR/kWh.
  2. A realistic intraday/imbalance uplift is included as a scenario, plus the
     break-even uplift as before.
  3. Sensitivities for the risk register actions (R02 TDTR curtailment,
     R04 grid tariff, R22 ancillary price decline) and a tornado chart.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# 1. INPUTS
# ---------------------------------------------------------------------------
POWER_MW = 300
LIFETIME = 15
TRADING_MW = 225            # 300 - 50 aFRR - 25 FCR

# A 4h battery earns more per MW in every energy market than a 2h one. KYOS's uplift
# estimate is for a 2h battery, so we scale it by the same ratio we found for
# day-ahead arbitrage in step 2 (EUR 131k vs 83k per MW). Without this, the 4h
# battery would be given the same uplift per MW as a 2h - unfair to the 4h.
UPLIFT_SCALE = {"2h": 1.0, "4h": 131 / 83}

# Equipment from the engineering workbook (BESS Project summary sheet)
DESIGNS = {
    "4h": {"containers": 234, "mwh_per_container": 6.431, "pcs_units": 78,
           # Year-1 revenues from step 3 (EUR million): arbitrage, aFRR, FCR
           "arbitrage": 29.47, "afrr": 11.23, "fcr": 3.12},
    "2h": {"containers": 142, "mwh_per_container": 6.012, "pcs_units": 71,
           "arbitrage": 18.76, "afrr": 11.23, "fcr": 3.12},
}

# Base-case assumptions. Each one can be changed by the sensitivity analysis.
BASE = {
    # Capex (2026 prices, EUR; 1 USD ~ 0.88 EUR)
    "dc_block_eur_per_kwh": 85,     # BYD MC Cube-T ~$74-94/kWh ex-works China (Coloprice, Sep 2026) + freight/duty to NL
    "pcs_skid_eur_m": 0.40,         # per Sungrow SC5000UD-MV skid (PCS adds ~$15-40/kWh, Coloprice) - ASSUMPTION
    "bop_eur_per_kwh": 35,          # civil, 33 kV, switchgear, SCADA, installation (Ember: EPC ~$50/kWh incl. grid) - ASSUMPTION
    "hv_connection_eur_m": 35,      # 2 x 200 MVA 380/33/33 kV transformers + 380 kV bays - ASSUMPTION
    "development_contingency": 0.10,
    "capex_factor": 1.00,           # used by the sensitivity analysis
    "opex_share": 0.015,            # of total capex per year
    # Grid (TenneT, 380 kV)
    "grid_tariff_eur_m_per_mw": 0.116,
    "tdtr_discount": 0.43,
    "tdtr_revenue_loss": 0.05,      # share of trading revenue lost to curtailment
    # Revenue
    # Intraday + imbalance uplift on top of day-ahead, EUR M per MW per year, for a 2-HOUR battery.
    # KYOS estimates ~EUR 200k/MW for a 2h battery; we take half (conservative).
    "intraday_uplift_eur_m_per_mw": 0.10,
    "arbitrage_decline": 0.02,
    "balancing_decline": 0.10,
    "capacity_fade": 0.02,
    "trading_fee": 0.07,
    # Finance
    "wacc": 0.08,
}


# ---------------------------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------------------------
def capex(design, a):
    """Capex breakdown in EUR million, built from the installed equipment."""
    d = DESIGNS[design]
    installed_mwh = d["containers"] * d["mwh_per_container"]
    parts = {
        "Battery DC blocks": installed_mwh * a["dc_block_eur_per_kwh"] / 1000,
        "PCS / MV skids": d["pcs_units"] * a["pcs_skid_eur_m"],
        "BOP and EPC": installed_mwh * a["bop_eur_per_kwh"] / 1000,
        "HV connection": a["hv_connection_eur_m"],
    }
    subtotal = sum(parts.values())
    parts["Development and contingency"] = subtotal * a["development_contingency"]
    return {k: v * a["capex_factor"] for k, v in parts.items()}, installed_mwh


def cash_flows(design, a):
    d = DESIGNS[design]
    parts, _ = capex(design, a)
    total_capex = sum(parts.values())
    flows = [-total_capex]
    for year in range(1, LIFETIME + 1):
        n = year - 1
        uplift = a["intraday_uplift_eur_m_per_mw"] * UPLIFT_SCALE[design] * TRADING_MW
        trading = (d["arbitrage"] + uplift) \
            * (1 - a["tdtr_revenue_loss"]) \
            * (1 - a["arbitrage_decline"]) ** n * (1 - a["capacity_fade"]) ** n
        balancing = (d["afrr"] + d["fcr"]) * (1 - a["balancing_decline"]) ** n
        revenue = (trading + balancing) * (1 - a["trading_fee"])
        grid = a["grid_tariff_eur_m_per_mw"] * POWER_MW * (1 - a["tdtr_discount"])
        opex = a["opex_share"] * total_capex
        flows.append(revenue - grid - opex)
    return flows, total_capex


def npv(rate, flows):
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(flows))


def irr(flows):
    try:
        return brentq(lambda r: npv(r, flows), -0.9, 1.0)
    except ValueError:
        return np.nan


def breakeven_uplift(design, a):
    """2h-equivalent uplift (EUR k/MW/yr) that makes NPV exactly 0. NPV is linear in the uplift."""
    lo, hi = dict(a, intraday_uplift_eur_m_per_mw=0.0), dict(a, intraday_uplift_eur_m_per_mw=0.1)
    n0, n1 = npv(a["wacc"], cash_flows(design, lo)[0]), npv(a["wacc"], cash_flows(design, hi)[0])
    return -n0 / (n1 - n0) * 100      # 0.1 EUR M/MW = 100 k/MW


# ---------------------------------------------------------------------------
# 3. RESULTS: capex and base case for both designs
# ---------------------------------------------------------------------------
rows = []
for design in DESIGNS:
    parts, installed = capex(design, BASE)
    flows, total = cash_flows(design, BASE)
    rows.append({"Design": design,
                 "Installed DC MWh": installed,
                 **{k: v for k, v in parts.items()},
                 "Total capex": total,
                 "EUR/kWh (rated)": total / (POWER_MW * int(design[0])) * 1000,
                 "Year-1 cash flow": flows[1],
                 "NPV @8%": npv(BASE["wacc"], flows),
                 "IRR (%)": irr(flows) * 100,
                 "Break-even uplift (k/MW, 2h-equiv.)": breakeven_uplift(design, BASE)})
results = pd.DataFrame(rows).set_index("Design").T
pd.set_option("display.width", 120)
print("Base case (EUR million unless stated), TDTR contract, uplift EUR 100k/MW (2h-equivalent):\n")
print(results.round(2).to_string())

# ---------------------------------------------------------------------------
# 4. SENSITIVITIES (4h design) -> tornado chart
# ---------------------------------------------------------------------------
# (label, parameter, low value, high value). Linked risk in brackets.
SENS = [
    ("Intraday uplift 50k / 150k per MW (R21)", "intraday_uplift_eur_m_per_mw", 0.05, 0.15),
    ("Capex +15% / -15% (R19, R20)", "capex_factor", 1.15, 0.85),
    ("Grid tariff +50% / -25% (R04)", "grid_tariff_eur_m_per_mw", 0.116 * 1.5, 0.116 * 0.75),
    ("WACC 9% / 7%", "wacc", 0.09, 0.07),
    ("TDTR curtailment loss 15% / 0% (R02)", "tdtr_revenue_loss", 0.15, 0.0),
    ("Balancing price decline -20% / -5% per yr (R22)", "balancing_decline", 0.20, 0.05),
    ("Capacity fade 3% / 1% per yr (R23)", "capacity_fade", 0.03, 0.01),
]

design = "4h"
base_npv = npv(BASE["wacc"], cash_flows(design, BASE)[0])
tornado = []
for label, key, low, high in SENS:
    out = []
    for value in (low, high):
        a = dict(BASE, **{key: value})
        out.append(npv(a["wacc"], cash_flows(design, a)[0]))
    tornado.append((label, out[0], out[1]))
tornado.sort(key=lambda t: abs(t[2] - t[1]))   # biggest swing ends up on top

print(f"\nSensitivity, 4h design (base NPV EUR {base_npv:.0f} M):")
for label, lo, hi in reversed(tornado):
    print(f"  {label:50s} {lo:7.0f} -> {hi:7.0f}")

fig, ax = plt.subplots(figsize=(10, 5))
for i, (label, lo, hi) in enumerate(tornado):
    ax.barh(i, lo - base_npv, left=base_npv, color="#c0504d")
    ax.barh(i, hi - base_npv, left=base_npv, color="#4f81bd")
ax.axvline(base_npv, color="black", linewidth=1)
ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_yticks(range(len(tornado)))
ax.set_yticklabels([t[0] for t in tornado])
ax.set_xlabel("NPV at 8% WACC (EUR million)")
ax.set_title("What moves the NPV? 300 MW / 1,200 MWh BESS, TDTR contract\n(red = downside case, blue = upside case, black line = base case)")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()