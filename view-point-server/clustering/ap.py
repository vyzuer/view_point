from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import numpy as np


def cluster(dump_path, file_name):

    ##############################################################################
    data = np.loadtxt(file_name)
    print data.shape
    
    # X = StandardScaler().fit_transform(data)
    # X = normalize(X, norm='l2')

    ##############################################################################
    af = AffinityPropagation(damping=0.9999, convergence_iter=50, max_iter=1000).fit(data)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    
    n_clusters = len(cluster_centers_indices)
    
    print('Estimated number of clusters: %d' % n_clusters)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
    
    label_file = dump_path + "labels.list"
    fp = open(label_file, 'w')
    for i in labels:
        fp.write("%d\n" % i)
    fp.close()

    num_cluster_file = dump_path + "_num_clusters.info"
    np.savetxt(num_cluster_file, n_clusters)

