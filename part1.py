
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


# TASK 1.2: HYPOTHESIS TESTING

# Hypothesised population mean
mu_0 = 8.0

# Calculate Z-statistic
z_stat = (mean_value - mu_0) / se_value

# Make decision at 5% significance level
if abs(z_stat) > 1.96:
    reject_h0 = "yes"
else:
    reject_h0 = "no"

# Display results
print("\nTASK 1.2 RESULTS")
print("-------------------------")
print("z_stat:", z_stat)
print("reject_h0:", reject_h0)

# Append results to output file
with open("outputs/part1_results.txt", "a") as file:
    file.write(f"z_stat={z_stat:.6f}\n")
    file.write(f"reject_h0={reject_h0}\n")

# TASK 1.3: SAMPLE SIZE CALCULATION

# Given values
Z = 1.96
E = 0.5

# Calculate required sample size
n_required = math.ceil((Z * sd_value / E) ** 2)

# Display results
print("\nTASK 1.3 RESULTS")
print("-------------------------")
print("n_required:", n_required)

# Append results to output file
with open("outputs/part1_results.txt", "a") as file:
    file.write(f"n_required={n_required}\n")
