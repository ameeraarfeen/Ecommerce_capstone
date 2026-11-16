
import csv
import numpy as np


def load_split_data(file_path="data/records.csv"):
    """Load the dataset, create the stratified split, and standardise predictors."""

    with open(file_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    feature_names = [
        "customer_tenure_months",
        "complaint_count_90d",
        "discount_order_share",
        "avg_order_value_inr_hundreds",
        "loyalty_score",
        "cod_flag",
        "plus_member_flag"
    ]

    X = np.array(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=float
    )

    y = np.array(
        [int(row["churned"]) for row in rows],
        dtype=int
    )

    # Stratified 70:30 split, preserving the original row order.
    train_indices = []
    test_indices = []

    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    cutoff_0 = (7 * len(class_0_indices)) // 10
    cutoff_1 = (7 * len(class_1_indices)) // 10

    train_0 = class_0_indices[:cutoff_0]
    test_0 = class_0_indices[cutoff_0:]

    train_1 = class_1_indices[:cutoff_1]
    test_1 = class_1_indices[cutoff_1:]

    train_indices = np.sort(
        np.concatenate((train_0, train_1))
    )

    test_indices = np.sort(
        np.concatenate((test_0, test_1))
    )
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # Fit standardisation parameters using training data only.
    train_mean = np.mean(X_train, axis=0)
    train_sd = np.std(X_train, axis=0, ddof=1)

    X_train_scaled = (X_train - train_mean) / train_sd
    X_test_scaled = (X_test - train_mean) / train_sd

    return (
        X, y, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, train_mean, train_sd
    )


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, probabilities):
    eps = 1e-15
    probabilities = np.clip(probabilities, eps, 1 - eps)

    return -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )


def train_logistic(X_train, y_train, learning_rate=0.1, iterations=1000):
    """Train logistic regression using full-batch gradient descent."""

    weights = np.zeros(X_train.shape[1])
    bias = 0.0
    loss_values = {}

    for epoch in range(1, iterations + 1):
        probabilities = sigmoid(X_train @ weights + bias)

        error = probabilities - y_train
        dw = (X_train.T @ error) / len(y_train)
        db = np.mean(error)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        if epoch in [1, 10, 100, 1000]:
            updated_probabilities = sigmoid(X_train @ weights + bias)
            loss_values[epoch] = binary_cross_entropy(
                y_train, updated_probabilities
            )

    return weights, bias, loss_values


def knn_probabilities(X_reference, y_reference, X_query, K):
    """Return the proportion of positive labels among the K nearest neighbours."""

    probabilities = []

    for point in X_query:
        distances = np.sqrt(np.sum((X_reference - point) ** 2, axis=1))
        nearest_indices = np.argsort(distances, kind="stable")[:K]

        positive_votes = np.sum(y_reference[nearest_indices] == 1)
        probabilities.append(positive_votes / K)

    return np.array(probabilities)


def knn_predict(X_reference, y_reference, X_query, K):
    """Predict using the majority vote of the K nearest neighbours."""

    predictions = []

    for point in X_query:
        distances = np.sqrt(np.sum((X_reference - point) ** 2, axis=1))
        nearest_indices = np.argsort(distances, kind="stable")[:K]

        positive_votes = np.sum(y_reference[nearest_indices] == 1)
        required_votes = (K + 1) // 2

        predictions.append(int(positive_votes >= required_votes))

    return np.array(predictions)


def select_best_k(X_train, y_train):
    """Select K using the same five-fold CV procedure as Part 2."""

    fold_ids = np.arange(len(y_train)) % 5
    k_values = list(range(1, 22, 2))

    results = []

    for K in k_values:
        fold_f1_scores = []

        for fold in range(5):
            validation_mask = fold_ids == fold
            training_mask = ~validation_mask

            X_fold_train = X_train[training_mask]
            y_fold_train = y_train[training_mask]

            X_fold_valid = X_train[validation_mask]
            y_fold_valid = y_train[validation_mask]

            predictions = knn_predict(
                X_fold_train, y_fold_train, X_fold_valid, K
            )

            metrics = calculate_metrics(y_fold_valid, predictions)
            fold_f1_scores.append(metrics["f1"])

        results.append((K, np.mean(fold_f1_scores)))

    best_K = max(results, key=lambda item: item[1])[0]

    return best_K, results


def calculate_metrics(y_true, predictions):
    """Calculate binary classification metrics from scratch."""

    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0

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
