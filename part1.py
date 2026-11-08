
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

# Extract independent variable X4
X4 = np.array(
    [float(row["avg_order_value_inr_hundreds"]) for row in rows]
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

# TASK 1.4: CLOSED-FORM SIMPLE LINEAR REGRESSION

# 1. Calculate means of X and Y
mean_X = np.mean(X4)
mean_Y = np.mean(value)

# 2. Calculate covariance and variance manually
cov_XY = np.sum((X4 - mean_X) * (value - mean_Y)) / (n - 1)

var_X = np.sum((X4 - mean_X) ** 2) / (n - 1)

# 3. Calculate slope
reg_slope = cov_XY / var_X

# 4. Calculate intercept
reg_intercept = mean_Y - (reg_slope * mean_X)

# 5. Calculate fitted values (predictions)
fitted_values = reg_intercept + (reg_slope * X4)

# 6. Calculate residuals
residuals = value - fitted_values

# REGRESSION PERFORMANCE METRICS

# 7. Total Sum of Squares
tss = np.sum((value - mean_Y) ** 2)

# 8. Explained Sum of Squares
ess = np.sum((fitted_values - mean_Y) ** 2)

# 9. Residual Sum of Squares
rss = np.sum(residuals ** 2)

# 10. Check TSS = ESS + RSS
if abs(tss - ess - rss) < (1e-6 * tss):
    tss_equals_ess_plus_rss = "yes"
else:
    tss_equals_ess_plus_rss = "no"

# 11. R-squared
r2 = 1 - (rss / tss)

# 12. Root Mean Squared Error
rmse = np.sqrt(rss / n)

# 13. Mean Absolute Error
mae = np.mean(np.abs(residuals))

# DISPLAY RESULTS

print("\nTASK 1.4 RESULTS")
print("-------------------------")

print("reg_slope:", reg_slope)
print("reg_intercept:", reg_intercept)
print("tss:", tss)
print("ess:", ess)
print("rss:", rss)
print("tss_equals_ess_plus_rss:", tss_equals_ess_plus_rss)
print("r2:", r2)
print("rmse:", rmse)
print("mae:", mae)

# APPEND RESULTS TO OUTPUT FILE

with open("outputs/part1_results.txt", "a") as file:
    file.write(f"reg_slope={reg_slope:.6f}\n")
    file.write(f"reg_intercept={reg_intercept:.6f}\n")
    file.write(f"tss={tss:.6f}\n")
    file.write(f"ess={ess:.6f}\n")
    file.write(f"rss={rss:.6f}\n")
    file.write(f"tss_equals_ess_plus_rss={tss_equals_ess_plus_rss}\n")
    file.write(f"r2={r2:.6f}\n")
    file.write(f"rmse={rmse:.6f}\n")
    file.write(f"mae={mae:.6f}\n")

# TASK 1.5: GRADIENT DESCENT

# 1. Standardize X4 using sample standard deviation
mean_X4 = np.mean(X4)
sd_X4 = np.std(X4, ddof=1)

z = (X4 - mean_X4) / sd_X4

# 2. Initialize parameters
m = 0.0
c = 0.0

# 3. Set learning rate and number of iterations
learning_rate = 0.1
iterations = 500

# 4. Calculate initial cost (before any update)
y_pred = m * z + c
errors = value - y_pred

gd_J0 = np.mean(errors ** 2)

# 5. Perform gradient descent
for i in range(1, iterations + 1):

    # Calculate predictions
    y_pred = m * z + c

    # Calculate errors
    errors = value - y_pred

    # Calculate gradients
    dm = (-2 / n) * np.sum(z * errors)
    dc = (-2 / n) * np.sum(errors)

    # Update slope and intercept
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    # Calculate cost after the update
    y_pred = m * z + c
    errors = value - y_pred
    cost = np.mean(errors ** 2)

    # Store cost at required checkpoints
    if i == 1:
        gd_J1 = cost
    elif i == 10:
        gd_J10 = cost
    elif i == 100:
        gd_J100 = cost
    elif i == 500:
        gd_J500 = cost

# 6. Convert coefficients back to original X4 scale
gd_slope = m / sd_X4

gd_intercept = c - (m * mean_X4 / sd_X4)

# 7. Compare with closed-form regression slope
gd_slope_gap = abs(gd_slope - reg_slope)

# DISPLAY RESULTS

print("\nTASK 1.5 RESULTS")
print("-------------------------")

print("gd_J0:", gd_J0)
print("gd_J1:", gd_J1)
print("gd_J10:", gd_J10)
print("gd_J100:", gd_J100)
print("gd_J500:", gd_J500)
print("gd_slope:", gd_slope)
print("gd_intercept:", gd_intercept)
print("gd_slope_gap:", gd_slope_gap)

# APPEND RESULTS TO OUTPUT FILE

with open("outputs/part1_results.txt", "a") as file:
    file.write(f"gd_J0={gd_J0:.6f}\n")
    file.write(f"gd_J1={gd_J1:.6f}\n")
    file.write(f"gd_J10={gd_J10:.6f}\n")
    file.write(f"gd_J100={gd_J100:.6f}\n")
    file.write(f"gd_J500={gd_J500:.6f}\n")
    file.write(f"gd_slope={gd_slope:.6f}\n")
    file.write(f"gd_intercept={gd_intercept:.6f}\n")
    file.write(f"gd_slope_gap={gd_slope_gap:.6f}\n")
