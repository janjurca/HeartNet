import numpy as np


def getClusterSums(vector):
    inCluster = False
    clusters = []
    curentSum = 0
    for val in vector:
        if val:
            inCluster = True
            curentSum += val
        if not val and inCluster:
            clusters.append(curentSum)
            inCluster = False
            curentSum = 0
    if inCluster:
        clusters.append(curentSum)
    return clusters


def bestClusterCoords(vector, sums):
    max_cluster_value = max(sums)
    cluster_index = sums.index(max_cluster_value)
    inCluster = False
    start_index, end_index, atcluster = 0, 0, 0
    for i, val in enumerate(vector):
        if val and not inCluster:
            inCluster = True
            if atcluster == cluster_index:
                start_index = i
        if not val and inCluster:
            inCluster = False
            if atcluster == cluster_index:
                end_index = i
                break
            atcluster += 1
    if inCluster:
        end_index = i
    return start_index, end_index


def getBoundingBox(mask, threshold=0.1):
    min_val = mask.shape[0] * threshold

    def clear(x):
        for i, val in enumerate(x):
            x[i] = val if val > min_val else 0
        return x
    print(min_val)
    x = np.sum(list(map(clear, np.sum(mask, 0).tolist())), 0)
    y = np.sum(list(map(clear, np.sum(mask, 2).tolist())), 0)
    z = np.sum(list(map(clear, np.sum(mask, 1).tolist())), 1)

    x_start_index, x_end_index = bestClusterCoords(x, getClusterSums(x))
    y_start_index, y_end_index = bestClusterCoords(y, getClusterSums(y))
    z_start_index, z_end_index = bestClusterCoords(z, getClusterSums(z))

    return x_start_index, x_end_index, y_start_index, y_end_index, z_start_index, z_end_index
