# Q4: Hierarchical Clustering without ML libraries

import math
import pandas as pd

# Dataset
data = {
    'Item': ['Apple', 'Banana', 'Orange', 'Grapes', 'Mango'],
    'Weight': [180, 120, 160, 5, 200],
    'Sugar': [10, 12, 9, 16, 14]
}

df = pd.DataFrame(data)

# Compute Euclidean distance between two items
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Create distance matrix
def create_distance_matrix(df):
    matrix = {}
    for i in range(len(df)):
        matrix[df['Item'][i]] = {}
        for j in range(len(df)):
            if i == j:
                matrix[df['Item'][i]][df['Item'][j]] = 0
            else:
                dist = euclidean_distance(
                    (df['Weight'][i], df['Sugar'][i]),
                    (df['Weight'][j], df['Sugar'][j])
                )
                matrix[df['Item'][i]][df['Item'][j]] = round(dist, 2)
    return matrix

# Hierarchical Agglomerative Clustering
def hierarchical_clustering(df):
    clusters = [[item] for item in df['Item']]
    distances = create_distance_matrix(df)

    print("Initial Clusters:", clusters)

    while len(clusters) > 1:
        min_dist = float('inf')
        pair_to_merge = (None, None)

        # Find closest pair of clusters
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dists = []
                for item1 in clusters[i]:
                    for item2 in clusters[j]:
                        dists.append(distances[item1][item2])
                avg_dist = sum(dists) / len(dists)
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    pair_to_merge = (i, j)

        # Merge clusters
        new_cluster = clusters[pair_to_merge[0]] + clusters[pair_to_merge[1]]
        clusters.pop(pair_to_merge[1])
        clusters.pop(pair_to_merge[0])
        clusters.append(new_cluster)

        print("\nMerged Clusters:", new_cluster)
        print("Remaining Clusters:", clusters)
        print("Distance of merge:", round(min_dist, 2))

    return clusters

# Run clustering
final_clusters = hierarchical_clustering(df)
print("\n✅ Final Cluster Formed:", final_clusters)
