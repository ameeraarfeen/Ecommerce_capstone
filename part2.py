import numpy as np
import pandas as pd
import os
import csv

# 1. Load the dataset
df = pd.read_csv("data/records.csv")

# 2. Select the seven input features
feature_columns = [
    "customer_tenure_months",       # X1
    "complaint_count_90d",          # X2
    "discount_order_share",        # X3
    "avg_order_value_inr_hundreds", # X4
    "loyalty_score",                # X5
    "cod_flag",                     # X6
    "plus_member_flag"              # X7
]

X = df[feature_columns].to_numpy(dtype=float)
y = df["churned"].to_numpy(dtype=int)

# 3. Separate rows according to outcome class
class_0_indices = np.where(y == 0)[0]
class_1_indices = np.where(y == 1)[0]

# 4. Calculate training cutoffs using integer arithmetic
cutoff_0 = (7 * len(class_0_indices)) // 10
cutoff_1 = (7 * len(class_1_indices)) // 10

# 5. Split each class into training and testing indices
train_0 = class_0_indices[:cutoff_0]
test_0 = class_0_indices[cutoff_0:]

train_1 = class_1_indices[:cutoff_1]
test_1 = class_1_indices[cutoff_1:]

# 6. Combine indices and preserve original file order
train_indices = np.sort(np.concatenate((train_0, train_1)))
test_indices = np.sort(np.concatenate((test_0, test_1)))

# 7. Create training and testing datasets
X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

# 8. Calculate split statistics
train_rows = len(X_train)
train_positives = int(np.sum(y_train == 1))

test_rows = len(X_test)
test_positives = int(np.sum(y_test == 1))

# 9. Calculate mean and sample SD using training data only
train_mean = np.mean(X_train, axis=0)
train_sd = np.std(X_train, axis=0, ddof=1)

# 10. Standardise training and testing data
X_train_scaled = (X_train - train_mean) / train_sd
X_test_scaled = (X_test - train_mean) / train_sd

# 11. Display results
print("TASK 2.1 RESULTS")
print("-------------------------")
print("train_rows =", train_rows)
print("train_positives =", train_positives)
print("test_rows =", test_rows)
print("test_positives =", test_positives)

print("\nTraining scaled shape:", X_train_scaled.shape)
print("Testing scaled shape:", X_test_scaled.shape)

print("\nTraining feature means:")
print(np.mean(X_train_scaled, axis=0))

print("\nTraining feature sample SDs:")
print(np.std(X_train_scaled, axis=0, ddof=1))

os.makedirs("outputs", exist_ok=True)

with open("outputs/part2_results.txt", "w") as file:
    file.write(f"train_rows={train_rows}\n")
    file.write(f"train_positives={train_positives}\n")
    file.write(f"test_rows={test_rows}\n")
    file.write(f"test_positives={test_positives}\n")

# PART 2.2 — LOGISTIC REGRESSION FROM SCRATCH

# STEP 1: SIGMOID FUNCTION
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# STEP 2: BINARY CROSS-ENTROPY LOSS

def binary_cross_entropy(y_true, probabilities):

    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

    loss = -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )

    return loss

# STEP 3: INITIALISE PARAMETERS

n = len(X_train_scaled)

weights = np.zeros(X_train_scaled.shape[1])
bias = 0.0

learning_rate = 0.1
iterations = 1000

# STEP 4: CALCULATE INITIAL LOSS (EPOCH 0)

initial_z = X_train_scaled @ weights + bias

initial_probabilities = sigmoid(initial_z)

loss_epoch_0 = binary_cross_entropy(
    y_train,
    initial_probabilities
)

# STEP 5: TRAIN USING FULL-BATCH GRADIENT DESCENT

loss_values = {}

for epoch in range(1, iterations + 1):

    # Calculate linear combination
    z = X_train_scaled @ weights + bias

    # Calculate predicted probabilities
    probabilities = sigmoid(z)

    # Calculate prediction errors
    errors = probabilities - y_train

    # Calculate gradients
    dw = (X_train_scaled.T @ errors) / n
    db = np.sum(errors) / n

    # Update weights and bias
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    # Calculate updated training loss
    updated_z = X_train_scaled @ weights + bias
    updated_probabilities = sigmoid(updated_z)

    loss = binary_cross_entropy(
        y_train,
        updated_probabilities
    )

    # Store required epoch checkpoints
    if epoch in [1, 10, 100, 1000]:
        loss_values[epoch] = loss

# STEP 6: EXTRACT LOSS CHECKPOINTS

loss_epoch_1 = loss_values[1]
loss_epoch_10 = loss_values[10]
loss_epoch_100 = loss_values[100]
loss_epoch_1000 = loss_values[1000]

# STEP 7: EVALUATE MODEL ON TEST DATA

test_z = X_test_scaled @ weights + bias

test_probabilities = sigmoid(test_z)

test_bce = binary_cross_entropy(
    y_test,
    test_probabilities
)

# STEP 8: DISPLAY RESULTS

print("PART 2.2 — LOGISTIC REGRESSION RESULTS")
print("=" * 45)

print(f"loss_epoch_0={loss_epoch_0:.10f}")
print(f"loss_epoch_1={loss_epoch_1:.10f}")
print(f"loss_epoch_10={loss_epoch_10:.10f}")
print(f"loss_epoch_100={loss_epoch_100:.10f}")
print(f"loss_epoch_1000={loss_epoch_1000:.10f}")

print("\nLEARNED WEIGHTS")
print("-" * 30)

for i, weight in enumerate(weights, start=1):
    print(f"weight_X{i}={weight:.10f}")

print(f"\nbias={bias:.10f}")
print(f"test_bce={test_bce:.10f}")

# STEP 9: SAVE RESULTS

os.makedirs("outputs", exist_ok=True)

with open("outputs/part2_results.txt", "a") as file:

    file.write("\n# Task 2.2 — Logistic Regression\n")

    file.write(f"loss_epoch_0={loss_epoch_0:.10f}\n")
    file.write(f"loss_epoch_1={loss_epoch_1:.10f}\n")
    file.write(f"loss_epoch_10={loss_epoch_10:.10f}\n")
    file.write(f"loss_epoch_100={loss_epoch_100:.10f}\n")
    file.write(f"loss_epoch_1000={loss_epoch_1000:.10f}\n")

    for i, weight in enumerate(weights, start=1):
        file.write(f"weight_X{i}={weight:.10f}\n")

    file.write(f"bias={bias:.10f}\n")
    file.write(f"test_bce={test_bce:.10f}\n")

print("\nTask 2.2 results saved to outputs/part2_results.txt")

# PART 2.3 — EFFECT OF LEARNING RATE

import numpy as np
import os

# STEP 1: DEFINE REQUIRED FUNCTIONS

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, probabilities):
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

    loss = -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )

    return loss

# STEP 2: DEFINE LEARNING RATES

learning_rates = [0.01, 0.1, 1.0]

iterations = 200
n = len(X_train_scaled)

learning_rate_results = {}

# STEP 3: TRAIN A SEPARATE MODEL FOR EACH LEARNING RATE

for lr in learning_rates:

    # Restart weights and bias for every model
    weights_lr = np.zeros(X_train_scaled.shape[1])
    bias_lr = 0.0

    # Perform 200 full-batch gradient descent updates
    for epoch in range(1, iterations + 1):

        # Calculate linear combination
        z = X_train_scaled @ weights_lr + bias_lr

        # Calculate predicted probabilities
        probabilities = sigmoid(z)

        # Calculate errors
        errors = probabilities - y_train

        # Calculate gradients
        dw = (X_train_scaled.T @ errors) / n
        db = np.sum(errors) / n

        # Update weights and bias
        weights_lr = weights_lr - lr * dw
        bias_lr = bias_lr - lr * db

    # Calculate final training loss after 200 updates
    final_probabilities = sigmoid(
        X_train_scaled @ weights_lr + bias_lr
    )

    final_loss = binary_cross_entropy(
        y_train,
        final_probabilities
    )

    learning_rate_results[lr] = final_loss


# STEP 4: DISPLAY RESULTS

print("PART 2.3 — LEARNING RATE COMPARISON")
print("=" * 45)

for lr, loss in learning_rate_results.items():
    print(f"learning_rate_{lr}_loss_200={loss:.10f}")


# STEP 5: SAVE RESULTS

os.makedirs("outputs", exist_ok=True)

with open("outputs/part2_results.txt", "a") as file:

    file.write("\n# Task 2.3 — Learning Rate Comparison\n")

    for lr, loss in learning_rate_results.items():
        file.write(f"learning_rate_{lr}_loss_200={loss:.10f}\n")

print("\nTask 2.3 results saved successfully.")

# TASK 2.4 — ODDS RATIOS

odds_ratios = np.exp(weights)

feature_names = [
    "X1", "X2", "X3", "X4", "X5", "X6", "X7"
]

odds_ratio_results = {}

print("\nPART 2.4 — ODDS RATIOS")
print("=" * 45)

for name, odds_ratio in zip(feature_names, odds_ratios):
    odds_ratio_results[name] = odds_ratio

    print(f"odds_ratio_{name}={odds_ratio:.10f}")

# TASK 2.5 — CONFUSION MATRIX AND METRICS

def calculate_metrics(y_true, y_pred):

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)

    accuracy = (tp + tn) / total

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }


# Evaluate at threshold 0.5
predictions_05 = (test_probabilities >= 0.5).astype(int)
metrics_05 = calculate_metrics(y_test, predictions_05)

# Evaluate at threshold 0.3
predictions_03 = (test_probabilities >= 0.3).astype(int)
metrics_03 = calculate_metrics(y_test, predictions_03)

# Blind accuracy: predict every customer as 0
blind_accuracy = np.mean(y_test == 0)

# Display results
print("\nPART 2.5 — CLASSIFICATION METRICS")
print("=" * 50)

for threshold, metrics in [
    (0.5, metrics_05),
    (0.3, metrics_03)
]:
    print(f"\nTHRESHOLD = {threshold}")

    for key, value in metrics.items():
        print(f"{key}_{str(threshold).replace('.', '')}={value}")

print(f"\nblind_accuracy={blind_accuracy}")

# TASK 2.6 — THRESHOLD SWEEP

threshold_results = []

print("\nPART 2.6 — THRESHOLD SWEEP")
print("=" * 50)

for k in range(1, 10):

    tau = k / 10

    # Convert probabilities into binary predictions
    predictions = (test_probabilities >= tau).astype(int)

    # Calculate confusion matrix values
    tp = int(np.sum((y_test == 1) & (predictions == 1)))
    fp = int(np.sum((y_test == 0) & (predictions == 1)))
    tn = int(np.sum((y_test == 0) & (predictions == 0)))
    fn = int(np.sum((y_test == 1) & (predictions == 0)))

    # Calculate TPR and FPR
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    threshold_results.append([
        tau, tp, fp, tn, fn, tpr, fpr
    ])

    print(
        f"tau={tau:.1f}, "
        f"tp={tp}, fp={fp}, tn={tn}, fn={fn}, "
        f"tpr={tpr:.6f}, fpr={fpr:.6f}"
    )


# Save results to CSV
os.makedirs("outputs", exist_ok=True)

with open("outputs/part2_thresholds.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "tau", "tp", "fp", "tn", "fn", "tpr", "fpr"
    ])

    writer.writerows(threshold_results)


print("\nThreshold results saved to outputs/part2_thresholds.csv")