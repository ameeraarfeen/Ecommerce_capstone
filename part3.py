
import csv
import numpy as np
from pathlib import Path


# ============================================================
# PART 3 — SEGMENTATION AND DIMENSIONALITY REDUCTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "records.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

n = len(rows)

if n != 800:
    raise ValueError(f"Expected 800 rows, but found {n}")

# Five continuous predictors X1 to X5
feature_columns = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score"
]

X_original = np.array(
    [[float(row[col]) for col in feature_columns] for row in rows],
    dtype=float
)

# Outcome and order value
outcome = np.array(
    [float(row["churned"]) for row in rows],
    dtype=float
)

value = np.array(
    [float(row["order_value_inr_hundreds"]) for row in rows],
    dtype=float
)


# Dictionary containing all results
results = {}


# ============================================================
# TASK 3.1 — K-MEANS FROM SCRATCH
# ============================================================

# Calculate column means and sample standard deviations
feature_means = np.mean(X_original, axis=0)
feature_sds = np.std(X_original, axis=0, ddof=1)

# Standardise all 800 rows
X = (X_original - feature_means) / feature_sds


def kmeans_from_scratch(data, K):
    """
    K-means clustering using the pinned initial centroids.

    Returns:
        labels: final cluster assignment for each row
        centroids: final centroid positions
        wcss: within-cluster sum of squares
    """

    n_rows = data.shape[0]

    # Pinned starting centroid indices
    step = n_rows // K

    initial_indices = [
        i * step for i in range(K)
    ]

    centroids = data[initial_indices].copy()

    # -1 means that a row has not yet been assigned
    labels = np.full(n_rows, -1, dtype=int)

    assignment_passes = 0

    while True:

        assignment_passes += 1

        # Calculate squared Euclidean distances
        distances = np.sum(
            (data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # np.argmin chooses the lower cluster number in a tie
        new_labels = np.argmin(distances, axis=1)

        # Check whether any row changed its cluster
        unchanged = np.array_equal(new_labels, labels)

        labels = new_labels.copy()

        # Update centroids
        for cluster in range(K):

            members = data[labels == cluster]

            if len(members) > 0:
                centroids[cluster] = np.mean(members, axis=0)

        # Stop at the first assignment pass with no changes
        if unchanged:
            break

    # Calculate WCSS using final centroids
    wcss = 0.0

    for cluster in range(K):

        members = data[labels == cluster]

        if len(members) > 0:
            squared_distances = np.sum(
                (members - centroids[cluster]) ** 2,
                axis=1
            )

            wcss += np.sum(squared_distances)

    return labels, centroids, wcss, assignment_passes


# Run K-means for K = 2, 3, 4, 5, 6
wcss_values = {}
kmeans_labels = {}

for K in range(2, 7):

    labels, centroids, wcss, passes = kmeans_from_scratch(X, K)

    wcss_values[K] = wcss
    kmeans_labels[K] = labels

    results[f"wcss_k{K}"] = wcss

    print(
        f"K={K}, WCSS={wcss:.6f}, "
        f"assignment_passes={passes}"
    )


# ============================================================
# TASK 3.2 — CHOOSE K USING THE ELBOW RULE
# ============================================================

elbow_k = None

for K in [3, 4, 5]:

    current_drop = wcss_values[K] - wcss_values[K + 1]
    previous_drop = wcss_values[K - 1] - wcss_values[K]

    if current_drop < 0.5 * previous_drop:
        elbow_k = K
        break

if elbow_k is None:
    raise ValueError("No elbow K found using the specified rule.")

results["elbow_k"] = elbow_k

print("\nElbow K:", elbow_k)


# ============================================================
# TASK 3.3 — PROFILE THE K = 3 SEGMENTS
# ============================================================

# The assignment specifically requires K=3 for profiling
labels_k3 = kmeans_labels[3]

# Calculate mean X3 for each original cluster
mean_x3 = {}

for cluster in range(3):

    members = X_original[labels_k3 == cluster]

    mean_x3[cluster] = np.mean(members[:, 2])


# Order clusters by ascending mean X3
ordered_clusters = sorted(
    range(3),
    key=lambda cluster: mean_x3[cluster]
)

# Convert original cluster numbers into segment A, B and C
segment_names = ["A", "B", "C"]

segment_labels = np.empty(n, dtype=object)
segment_masks = {}

for name, cluster in zip(segment_names, ordered_clusters):

    mask = labels_k3 == cluster

    segment_labels[mask] = name
    segment_masks[name] = mask


# Create the segments CSV
segments_file = OUTPUT_DIR / "part3_segments.csv"

segment_header = [
    "segment",
    "size",
    "mean_X1",
    "mean_X2",
    "mean_X3",
    "mean_X4",
    "mean_X5",
    "outcome_rate",
    "mean_value"
]

with open(segments_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)
    writer.writerow(segment_header)

    for name in segment_names:

        mask = segment_masks[name]

        segment_data = X_original[mask]
        segment_size = int(np.sum(mask))

        segment_means = np.mean(segment_data, axis=0)

        outcome_rate = np.mean(outcome[mask])
        mean_value = np.mean(value[mask])

        results[f"size_segment_{name}"] = segment_size

        writer.writerow([
            name,
            segment_size,
            *[f"{v:.10f}" for v in segment_means],
            f"{outcome_rate:.10f}",
            f"{mean_value:.10f}"
        ])

print("\nSegment sizes:")

for name in segment_names:
    print(
        f"Segment {name}: "
        f"{results[f'size_segment_{name}']}"
    )


# ============================================================
# TASK 3.4 — HIERARCHICAL CLUSTERING
# ============================================================

# Supplied pairwise dissimilarity matrix
distance_matrix = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])

cohort_names = ["E1", "E2", "E3", "E4", "E5", "E6"]


def cluster_name(cluster):
    """
    Convert cluster member indices into names
    in ascending order, joined by +.
    """
    return "+".join(
        cohort_names[i]
        for i in sorted(cluster)
    )


def cluster_distance(cluster_a, cluster_b, linkage):
    """
    Calculate linkage distance between two clusters.
    """

    distances = []

    for i in cluster_a:
        for j in cluster_b:
            distances.append(distance_matrix[i, j])

    if linkage == "single":
        return min(distances)

    elif linkage == "complete":
        return max(distances)

    elif linkage == "average":
        return sum(distances) / len(distances)

    else:
        raise ValueError("Unknown linkage method")


def hierarchical_clustering(linkage):
    """
    Agglomerative hierarchical clustering.

    Starts with six individual clusters and performs
    five merges until one cluster remains.
    """

    clusters = [
        {i} for i in range(6)
    ]

    merge_records = []
    merge_heights = []

    for merge_no in range(1, 6):

        best_distance = float("inf")
        best_pair = None

        # Examine every pair of current clusters
        for i in range(len(clusters)):

            for j in range(i + 1, len(clusters)):

                distance = cluster_distance(
                    clusters[i],
                    clusters[j],
                    linkage
                )

                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        i, j = best_pair

        cluster_a = clusters[i]
        cluster_b = clusters[j]

        # Ensure cluster_a has the smaller minimum member
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a

        name_a = cluster_name(cluster_a)
        name_b = cluster_name(cluster_b)

        merge_records.append([
            linkage,
            merge_no,
            name_a,
            name_b,
            best_distance
        ])

        merge_heights.append(best_distance)

        # Merge the two clusters
        merged_cluster = cluster_a.union(cluster_b)

        # Remove old clusters and add the new cluster
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (i, j)
        ]

        clusters.append(merged_cluster)

        # Keep clusters ordered by their smallest member
        clusters.sort(key=lambda cluster: min(cluster))

    # Find the largest gap between consecutive merge heights
    gaps = np.diff(merge_heights)

    largest_gap_index = int(np.argmax(gaps))

    # Number of merges below the cut
    merges_below_cut = largest_gap_index + 1

    clusters_at_cut = 6 - merges_below_cut

    return merge_records, merge_heights, clusters_at_cut


linkage_methods = ["single", "complete", "average"]

all_hierarchical_records = []

for linkage in linkage_methods:

    records, heights, clusters_at_cut = hierarchical_clustering(linkage)

    all_hierarchical_records.extend(records)

    results[f"{linkage}_clusters_at_cut"] = clusters_at_cut

    print(f"\n{linkage.capitalize()} linkage")
    print("Merge heights:", heights)
    print("Clusters at cut:", clusters_at_cut)


# Save hierarchical clustering CSV
hierarchical_file = OUTPUT_DIR / "part3_hierarchical.csv"

with open(
    hierarchical_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "linkage",
        "merge_no",
        "cluster_a",
        "cluster_b",
        "height"
    ])

    for record in all_hierarchical_records:

        writer.writerow([
            record[0],
            record[1],
            record[2],
            record[3],
            f"{record[4]:.10f}"
        ])


# ============================================================
# TASK 3.5 — PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================

# Step 1: Mean vector after standardisation
pca_mean = np.mean(X, axis=0)

# Step 2: Centre the data
X_centered = X - pca_mean

# Step 3: Covariance matrix
covariance_matrix = (
    X_centered.T @ X_centered
) / (n - 1)

# Step 4: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Step 5: Sort eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 6: Apply the required eigenvector sign convention
for component in range(eigenvectors.shape[1]):

    largest_index = np.argmax(
        np.abs(eigenvectors[:, component])
    )

    if eigenvectors[largest_index, component] < 0:
        eigenvectors[:, component] *= -1

# Keep the top two eigenvectors
top_two_eigenvectors = eigenvectors[:, :2]

# Step 7: Project data onto the first two principal components
projections = X_centered @ top_two_eigenvectors


# Eigenvalue sum
eigenvalue_sum = np.sum(eigenvalues)

results["eigenvalue_sum"] = eigenvalue_sum

# Eigenvalues, explained variance ratios and cumulative ratios
explained_ratios = eigenvalues / eigenvalue_sum

cumulative_ratios = np.cumsum(explained_ratios)

for i in range(5):

    results[f"eigenvalue_{i + 1}"] = eigenvalues[i]

    results[f"explained_ratio_{i + 1}"] = explained_ratios[i]

    results[f"cumulative_ratio_{i + 1}"] = cumulative_ratios[i]


# Smallest number of components explaining at least 80%
components_for_80pct = int(
    np.argmax(cumulative_ratios >= 0.80) + 1
)

results["components_for_80pct"] = components_for_80pct


# PC1 loadings
for i in range(5):

    results[f"pc1_loading_X{i + 1}"] = top_two_eigenvectors[i, 0]

# PC2 loadings
for i in range(5):

    results[f"pc2_loading_X{i + 1}"] = top_two_eigenvectors[i, 1]


# Projections of the first three data rows
for row_index in range(3):

    results[f"row{row_index + 1}_pc1"] = projections[row_index, 0]

    results[f"row{row_index + 1}_pc2"] = projections[row_index, 1]


# Mean PC1 score for each K=3 segment
for name in segment_names:

    mask = segment_masks[name]

    mean_pc1 = np.mean(projections[mask, 0])

    results[f"mean_pc1_segment_{name}"] = mean_pc1


# ============================================================
# SAVE ALL RESULTS TO part3_results.txt
# ============================================================

results_file = OUTPUT_DIR / "part3_results.txt"

with open(results_file, "w", encoding="utf-8") as file:

    for key, val in results.items():

        if isinstance(val, (int, np.integer)):
            file.write(f"{key}={val}\n")

        else:
            file.write(f"{key}={val:.10f}\n")


# ============================================================
# FINAL CHECKS
# ============================================================

assert sum(
    results[f"size_segment_{name}"]
    for name in segment_names
) == 800

assert len(all_hierarchical_records) == 15

assert abs(results["eigenvalue_sum"] - 5.0) < 1e-8

print("\n========================================")
print("PART 3 COMPLETED SUCCESSFULLY")
print("========================================")

print("\nGenerated files:")
print("1. outputs/part3_results.txt")
print("2. outputs/part3_segments.csv")
print("3. outputs/part3_hierarchical.csv")

print("\nImportant checkpoints:")
print(f"wcss_k2 = {results['wcss_k2']:.4f}")
print(f"elbow_k = {results['elbow_k']}")
print(f"eigenvalue_sum = {results['eigenvalue_sum']:.4f}")
print(f"components_for_80pct = {results['components_for_80pct']}")
