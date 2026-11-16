"""
Builds flights_two_features.csv, the small dataset used by logistic_regression.py.

Source: the nycflights13 dataset, which records all flights that departed New York
City's three airports (JFK, LGA and EWR) in 2013.
Install once with:  pip install nycflights13

This script only prepares data. The model itself is in logistic_regression.py.
"""
import pandas as pd
from nycflights13 import flights

SEED = 42
N_SAMPLE = 1000          # small enough to see every point on a scatter plot
LATE_THRESHOLD = 15      # minutes; the usual industry on-time cut-off

# Keep flights that actually flew (cancelled flights have no delay values)
complete = flights.dropna(subset=["dep_delay", "arr_delay"])
sample = complete.sample(n=N_SAMPLE, random_state=SEED)

out = pd.DataFrame({
    # Feature 1: minutes between scheduled and actual departure (negative = early)
    "dep_delay_min": sample["dep_delay"].astype(int).values,
    # Feature 2: scheduled departure time as a decimal hour, e.g. 15:45 -> 15.75
    "sched_dep_hour": (sample["sched_dep_time"] // 100 + (sample["sched_dep_time"] % 100) / 60).round(2).values,
    # Kept for transparency: the target is derived from this column
    "arr_delay_min": sample["arr_delay"].astype(int).values,
})
# Target: 1 if the flight arrived 15 or more minutes late, otherwise 0
out["late_arrival"] = (out["arr_delay_min"] >= LATE_THRESHOLD).astype(int)
out.to_csv("flights_two_features.csv", index=False)

print(f"All flights: {len(flights)} | flown, with delay data: {len(complete)} | sampled: {len(out)}")
print(out["late_arrival"].value_counts().to_dict())
print(out.describe().round(2))
print(out.groupby("late_arrival")[["dep_delay_min", "sched_dep_hour"]].mean().round(2))
