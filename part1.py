
import numpy as np
import csv
import os
import math

# LOAD DATASET

data_path = "data/records.csv"

with open(data_path, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Extract target variable
value = np.array(
    [float(row["order_value_inr_hundreds"]) for row in rows]
)

# TASK 1.1: ESTIMATION WITH 95% CONFIDENCE INTERVAL-->

# 1. Calculate sample size
n = len(value)

# 2. Calculate mean
mean_value = np.mean(value)

# 3. Calculate sample standard deviation
sd_value = np.std(value, ddof=1)

# 4. Calculate standard error
se_value = sd_value / np.sqrt(n)

# 5. Calculate 95% confidence interval
ci95_low = mean_value - (1.96 * se_value)
ci95_high = mean_value + (1.96 * se_value)


print("\nPART 1.1 RESULTS")
print("-------------------------")

print("n:", n)
print("mean_value:", mean_value)
print("sd_value:", sd_value)
print("se_value:", se_value)
print("ci95_low:", ci95_low)
print("ci95_high:", ci95_high)

# SAVE RESULTS

os.makedirs("outputs", exist_ok=True)

with open("outputs/part1_results.txt", "w") as file:
    file.write(f"n={n}\n")
    file.write(f"mean_value={mean_value:.6f}\n")
    file.write(f"sd_value={sd_value:.6f}\n")
    file.write(f"se_value={se_value:.6f}\n")
    file.write(f"ci95_low={ci95_low:.6f}\n")
    file.write(f"ci95_high={ci95_high:.6f}\n")


