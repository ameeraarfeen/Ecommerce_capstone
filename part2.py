import numpy as np
import pandas as pd
import os

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


# task 2.2

