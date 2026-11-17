# -*- coding: utf-8 -*-
"""
BESS case study - Step 1: load and explore Dutch day-ahead prices (2025)
Data: Ember hourly wholesale prices (source: ENTSO-E Transparency Platform, TR 12.1.D)

"""

import pandas as pd               # tables of data ("DataFrames")
import matplotlib.pyplot as plt   # plotting

# --- 1. Load the CSV ----------------------------------------------------
# parse_dates tells pandas to read that column as real dates, not text
df = pd.read_csv("Netherlands.csv", parse_dates=["Datetime (Local)"])

# Keep only the two columns needed and given them short names
df = df[["Datetime (Local)", "Price (EUR/MWhe)"]]
df.columns = ["time", "price"]

# Use time as the index (the "row labels"), which makes filtering by date easy
df = df.set_index("time")

# --- 2. Keep only 2025 (the modelling year) ------------------------------
prices = df.loc["2025"]["price"]

print("Hours in 2025:", len(prices))            # expect 8760
print("Average price (EUR/MWh):", round(prices.mean(), 1))
print("Minimum price:", prices.min())
print("Maximum price:", prices.max())
print("Hours with negative prices:", (prices < 0).sum())

# --- 3. Daily spread: the battery's raw opportunity -----------------------
# For each day: highest hourly price minus lowest hourly price
daily_spread = prices.resample("D").max() - prices.resample("D").min()
print("Average daily max-min spread (EUR/MWh):", round(daily_spread.mean(), 1))

# --- 4. Plots ------------------------------------------------------------
# a) One example week, to see the daily up-and-down pattern
prices.loc["2025-06-02":"2025-06-08"].plot(figsize=(10, 4))
plt.title("Dutch day-ahead prices, one week in June 2025")
plt.ylabel("EUR/MWh")
plt.grid(alpha=0.3)
plt.show()

# b) Average price for each hour of the day (the "duck curve")
hourly_profile = prices.groupby(prices.index.hour).mean()
plt.figure(figsize=(10, 4))
plt.bar(hourly_profile.index, hourly_profile.values)
plt.title("Average price by hour of day, 2025")
plt.xlabel("Hour of day")
plt.ylabel("EUR/MWh")
plt.show()