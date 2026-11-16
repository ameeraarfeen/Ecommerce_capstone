
import csv
import math
import os
import numpy as np

from part2_helpers import (
    load_split_data,
    sigmoid,
    train_logistic,
    knn_probabilities,
    select_best_k,
    calculate_metrics
)


# ============================================================
# PART 4 — MODEL CHOICE AND ARCHITECTURE
# ============================================================

os.makedirs("outputs", exist_ok=True)

results = {}


# ============================================================
# TASK 4.1 — DECISION TREE SPLIT COMPARISON
# ============================================================

(
    X, y, X_train, X_test, y_train, y_test,
    X_train_scaled, X_test_scaled, train_mean, train_sd
) = load_split_data()


def entropy(labels):
    """Calculate binary entropy."""

    if len(labels) == 0:
        return 0.0

    probabilities = np.bincount(labels, minlength=2) / len(labels)

    value = 0.0

    for probability in probabilities:
        if probability > 0:
            value -= probability * math.log2(probability)

    return value


def gini(labels):
    """Calculate binary Gini impurity."""

    if len(labels) == 0:
        return 0.0

    probabilities = np.bincount(labels, minlength=2) / len(labels)

    return 1.0 - np.sum(probabilities ** 2)


root_entropy = entropy(y_train)
root_gini = gini(y_train)

# The four candidate splits use original, unstandardised predictors.
split_conditions = {
    "S1": X_train[:, 5] == 1,
    "S2": X_train[:, 6] == 1,
    "S3": X_train[:, 2] > 0.5,
    "S4": X_train[:, 0] <= 24
}

split_rows = []

for split_name, yes_mask in split_conditions.items():

    no_mask = ~yes_mask

    y_yes = y_train[yes_mask]
    y_no = y_train[no_mask]

    n_yes = len(y_yes)
    n_no = len(y_no)

    positives_yes = int(np.sum(y_yes == 1))
    positives_no = int(np.sum(y_no == 1))

    weighted_entropy = (
        (n_yes / len(y_train)) * entropy(y_yes)
        + (n_no / len(y_train)) * entropy(y_no)
    )

    weighted_gini = (
        (n_yes / len(y_train)) * gini(y_yes)
        + (n_no / len(y_train)) * gini(y_no)
    )

    information_gain = root_entropy - weighted_entropy
    gini_gain = root_gini - weighted_gini

    split_rows.append({
        "split": split_name,
        "n_yes": n_yes,
        "positives_yes": positives_yes,
        "n_no": n_no,
        "positives_no": positives_no,
        "info_gain": information_gain,
        "gini_gain": gini_gain
    })


best_info_split = max(
    split_rows,
    key=lambda row: row["info_gain"]
)

best_gini_split = max(
    split_rows,
    key=lambda row: row["gini_gain"]
)

results["root_entropy"] = root_entropy
results["root_gini"] = root_gini
results["best_split_by_info_gain"] = best_info_split["split"]
results["best_split_by_gini_gain"] = best_gini_split["split"]

with open(
    "outputs/part4_splits.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "split",
            "n_yes",
            "positives_yes",
            "n_no",
            "positives_no",
            "info_gain",
            "gini_gain"
        ]
    )

    writer.writeheader()

    for row in split_rows:
        writer.writerow(row)


# ============================================================
# TASK 4.2 — MODEL COMPARISON AND VOTING
# ============================================================

# Logistic regression using the same training procedure as Part 2.
logistic_weights, logistic_bias, logistic_losses = train_logistic(
    X_train_scaled,
    y_train,
    learning_rate=0.1,
    iterations=1000
)

logistic_probabilities = sigmoid(
    X_test_scaled @ logistic_weights + logistic_bias
)

logistic_predictions = (logistic_probabilities >= 0.3).astype(int)


# KNN: select K using the same five-fold CV procedure as Part 2.
best_K, knn_cv_results = select_best_k(
    X_train_scaled,
    y_train
)

knn_test_probabilities = knn_probabilities(
    X_train_scaled,
    y_train,
    X_test_scaled,
    best_K
)

required_votes = (best_K + 1) // 2

knn_predictions = (
    knn_test_probabilities >= required_votes / best_K
).astype(int)


# Decision stump using the best information-gain split.
best_split_name = best_info_split["split"]
stump_yes_mask = split_conditions[best_split_name]
stump_no_mask = ~stump_yes_mask

stump_rate_yes = np.mean(y_train[stump_yes_mask])
stump_rate_no = np.mean(y_train[stump_no_mask])

# Apply the same selected split condition to the test rows.
if best_split_name == "S1":
    test_yes_mask = X_test[:, 5] == 1
elif best_split_name == "S2":
    test_yes_mask = X_test[:, 6] == 1
elif best_split_name == "S3":
    test_yes_mask = X_test[:, 2] > 0.5
else:
    test_yes_mask = X_test[:, 0] <= 24

stump_test_probabilities = np.where(
    test_yes_mask,
    stump_rate_yes,
    stump_rate_no
)

stump_predictions = (
    stump_test_probabilities >= 0.3
).astype(int)


# Hard voting: at least two of three models predict positive.
hard_vote_count = (
    logistic_predictions
    + knn_predictions
    + stump_predictions
)

hard_vote_predictions = (hard_vote_count >= 2).astype(int)


# Soft voting: average the three model probabilities.
soft_vote_probabilities = (
    logistic_probabilities
    + knn_test_probabilities
    + stump_test_probabilities
) / 3.0

soft_vote_predictions = (
    soft_vote_probabilities >= 0.3
).astype(int)


models = {
    "logistic": logistic_predictions,
    "knn": knn_predictions,
    "stump": stump_predictions,
    "hard_vote": hard_vote_predictions,
    "soft_vote": soft_vote_predictions
}

voting_rows = []

for model_name, predictions in models.items():

    metrics = calculate_metrics(y_test, predictions)

    voting_rows.append({
        "model": model_name,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
    })


with open(
    "outputs/part4_voting.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "model",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "f1"
        ]
    )

    writer.writeheader()
    writer.writerows(voting_rows)


results["stump_rate_yes"] = stump_rate_yes
results["stump_rate_no"] = stump_rate_no
results["knn_best_K"] = best_K


# ============================================================
# TASK 4.3 — ADABOOST: TWO ROUNDS
# ============================================================

ada_y = np.array([1, 1, 1, -1, -1, -1, 1, -1])

weak_learners = [
    np.array([1, 1, -1, -1, -1, -1, -1, -1]),
    np.array([-1, 1, 1, -1, 1, -1, 1, -1])
]

sample_weights = np.full(len(ada_y), 1 / len(ada_y))

adaboost_rows = []
alphas = []

for round_number, predictions in enumerate(weak_learners, start=1):

    weights_before = sample_weights.copy()

    incorrect = predictions != ada_y

    error = np.sum(weights_before[incorrect])

    # Avoid division by zero in the alpha calculation.
    error = np.clip(error, 1e-15, 1 - 1e-15)

    alpha = 0.5 * math.log((1 - error) / error)

    unnormalised_weights = (
        weights_before
        * np.exp(-alpha * ada_y * predictions)
    )

    sample_weights = (
        unnormalised_weights / np.sum(unnormalised_weights)
    )

    alphas.append(alpha)

    for point_index in range(len(ada_y)):

        adaboost_rows.append({
            "round": round_number,
            "point": point_index + 1,
            "weight_before": weights_before[point_index],
            "weight_unnormalised": unnormalised_weights[point_index],
            "weight_after": sample_weights[point_index]
        })

    results[f"round{round_number}_error"] = error
    results[f"round{round_number}_alpha"] = alpha


# Final ensemble prediction from the weighted weak learners.
ensemble_score = (
    alphas[0] * weak_learners[0]
    + alphas[1] * weak_learners[1]
)

ensemble_predictions = np.where(
    ensemble_score >= 0,
    1,
    -1
)

ensemble_accuracy = np.mean(ensemble_predictions == ada_y)

results["ensemble_accuracy"] = ensemble_accuracy


with open(
    "outputs/part4_adaboost.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "round",
            "point",
            "weight_before",
            "weight_unnormalised",
            "weight_after"
        ]
    )

    writer.writeheader()
    writer.writerows(adaboost_rows)


# ============================================================
# TASK 4.4 — RANDOM FOREST OUT-OF-BAG FRACTION
# ============================================================

number_of_training_rows = len(y_train)

oob_fraction = (
    1 - 1 / number_of_training_rows
) ** number_of_training_rows

results["oob_fraction"] = oob_fraction


# ============================================================
# TASK 4.5 — NEURAL NETWORK PARAMETER COUNTS
# ============================================================

# MLP: 784 -> 16 -> 16 -> 10
mlp_hidden1 = 784 * 16 + 16
mlp_hidden2 = 16 * 16 + 16
mlp_output = 16 * 10 + 10

mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output


# CNN:
# Input: 28 x 28 x 1
# Conv1: 5 x 5, 6 filters, padding 2
# Pool1: 2 x 2, stride 2
# Conv2: 5 x 5, 16 filters, no padding
# Pool2: 2 x 2, stride 2
# Flatten: 5 x 5 x 16 = 400
# Dense: 400 -> 32 -> 10

cnn_conv1 = (5 * 5 * 1 + 1) * 6
cnn_conv2 = (5 * 5 * 6 + 1) * 16
cnn_dense1 = 400 * 32 + 32
cnn_dense2 = 32 * 10 + 10

cnn_total = (
    cnn_conv1
    + cnn_conv2
    + cnn_dense1
    + cnn_dense2
)


# Fully connected layer:
# Input: 128 x 128 x 3
# Output neurons: 1024
fc_layer_params = (128 * 128 * 3) * 1024 + 1024


# Convolutional layer:
# Kernel: 3 x 3
# Input channels: 3
# Filters: 64
conv_layer_params = (3 * 3 * 3) * 64 + 64


# RNN:
# Input size = 64
# Hidden size = 128
# Output size = 16
rnn_w_xh = 64 * 128
rnn_w_hh = 128 * 128
rnn_w_hy = 128 * 16
rnn_b_h = 128
rnn_b_y = 16

rnn_total = (
    rnn_w_xh
    + rnn_w_hh
    + rnn_w_hy
    + rnn_b_h
    + rnn_b_y
)


results["mlp_total"] = mlp_total
results["cnn_total"] = cnn_total
results["fc_layer_params"] = fc_layer_params
results["conv_layer_params"] = conv_layer_params
results["rnn_total"] = rnn_total


architecture_rows = [
    {"layer": "mlp_hidden1", "parameters": mlp_hidden1},
    {"layer": "mlp_hidden2", "parameters": mlp_hidden2},
    {"layer": "mlp_output", "parameters": mlp_output},
    {"layer": "cnn_conv1", "parameters": cnn_conv1},
    {"layer": "cnn_conv2", "parameters": cnn_conv2},
    {"layer": "cnn_dense1", "parameters": cnn_dense1},
    {"layer": "cnn_dense2", "parameters": cnn_dense2}
]

with open(
    "outputs/part4_architecture.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["layer", "parameters"]
    )

    writer.writeheader()
    writer.writerows(architecture_rows)


# ============================================================
# WRITE MAIN RESULTS FILE
# ============================================================

with open(
    "outputs/part4_results.txt",
    "w",
    encoding="utf-8"
) as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key}={value:.10f}\n")
        else:
            file.write(f"{key}={value}\n")


# ============================================================
# DISPLAY CHECKPOINTS
# ============================================================

print("PART 4 RESULTS")
print("=" * 50)

for key, value in results.items():
    print(f"{key}={value}")

print("\nGenerated files:")
print("outputs/part4_results.txt")
print("outputs/part4_splits.csv")
print("outputs/part4_voting.csv")
print("outputs/part4_adaboost.csv")
print("outputs/part4_architecture.csv")
