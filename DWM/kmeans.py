import math 

points = {
    'A1': (5, 5),
    'A2': (4, 5),
    'A3': (18, 4),
    'B1': (5, 8),
    'B2': (7, 1),
    'B3': (7, 4),
    'C1': (1, 2),
    'C2': (8, 9)
}

centroids = {
    'C1': (1, 2),
    'C2': (8, 9)
}


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2 )

def compute_mean(cluster_points):
    x_coords= [ p[0] for p in cluster_points]
    y_coords= [ p[1] for p in cluster_points]
    return (
        round(sum(x_coords)/len(x_coords), 2),
        round(sum(y_coords)/len(y_coords), 2)
    )

for i in range(10):
    clusters = {
        'C1':[], 
        'C2': []
    }

    for label in points:
        point=points[label]
        distances={}
        for c in centroids:
            d= euclidean_distance(point, centroids[c])
            distances[c]=d
        min_dist=float('inf')
        nearest=None
        for d in distances:
            if(distances[d]<min_dist):
                min_dist=distances[d]
                nearest=d

        clusters[nearest].append(point)

    new_centroids = {}
    for c in clusters:
        if len(clusters[c]) > 0:
            new_centroids[c] = compute_mean(clusters[c])
        else:
            new_centroids[c] = centroids[c] 

    print("\nIteration", i + 1)
    print("Clusters:", clusters)
    print("Centroids:", new_centroids)

    if new_centroids==centroids:
        print("Converged")
        break 

    centroids = new_centroids

print("\nFinal Clusters:")
for c in clusters:
    print(c, ":", clusters[c])







    