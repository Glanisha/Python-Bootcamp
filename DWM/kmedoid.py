import numpy as np 

X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
])

k = 2
medoids = np.array([
    [1.0, 2.0],
    [1.5, 1.8]
])

def manhattan_distance(p1, p2):
    dist = 0
    for i in range(len(p1)):
        dist += abs(p1[i] - p2[i])
    return dist

def assign_points(X, medoids):
    clusters = []
    for i in range(len(X)):
        dist = [manhattan_distance(X[i], medoid) for medoid in medoids]
        clusters.append(np.argmin(dist))
    return np.array(clusters)

def total_cost(X, medoids, clusters):
    cost = 0
    for i in range(len(X)):
        cost += manhattan_distance(X[i], medoids[clusters[i]])
    return cost

for iteration in range(10):
    clusters = assign_points(X, medoids)
    current_cost = total_cost(X, medoids, clusters)

    best_medoids = np.copy(medoids)
    best_cost = current_cost

    for i in range(k):
        for j in range(len(X)): 
            new_medoids = np.copy(medoids)
            new_medoids[i] = X[j]

            new_clusters = assign_points(X, new_medoids)
            new_cost = total_cost(X, new_medoids, new_clusters)

            if new_cost < best_cost:
                best_cost = new_cost
                best_medoids = new_medoids

    if np.allclose(medoids, best_medoids):
        print(f"Converged after {iteration+1} iterations.\n")
        break

    medoids = best_medoids

final_clusters = assign_points(X, medoids)

print("Final Medoids:\n", medoids)
print("\nCluster Assignments:", final_clusters)
