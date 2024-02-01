from sklearn.cluster import *

def paperClusterByDis(dis, name_papers, n_cluster, method='AG',linkage='average'):
    if method == 'AG':
        cmodel = AgglomerativeClustering(n_clusters=n_cluster, linkage= linkage, affinity='precomputed')
    elif method == 'AP':
        cmodel = AffinityPropagation(damping=0.5, affinity='precomputed')
    else:
        cmodel = DBSCAN(eps=0.25, min_samples=5, metric='precomputed')
    indexs = cmodel.fit_predict(dis)
    result = []

    separates = []
    for i, value in enumerate(indexs):
        if i >= len(name_papers):
            break

        if value == -1:
            separates.append(name_papers[i])
            continue

        while value >= len(result):
             result.append([])

        result[value].append(name_papers[i])

    if len(separates) > 0:
        result.append(separates)

    return result




if __name__ == '__main__':
    pass
