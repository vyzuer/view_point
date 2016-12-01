import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale, normalize


def cluster(dump_path, file_name):

    ##############################################################################
    data = np.loadtxt(file_name)
    print data.shape
    
    # X = StandardScaler().fit_transform(data)
    # X = normalize(X, norm='l2')

    ##############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.355, min_samples=10).fit(X)
    core_samples = db.core_sample_indices_
    components = db.components_
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print('Estimated number of clusters: %d' % n_clusters)
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels))
    
    label_file = dump_path + "labels.list"
    fp = open(label_file, 'w')
    for i in labels:
        fp.write("%d\n" % i)
    fp.close()

    num_cluster_file = dump_path + "_num_clusters.info"
    fp = open(num_cluster_file, 'w')
    fp.write("%d" % n_clusters)
    fp.close()

    centre_file = dump_path + "_centers.info"
    np.savetxt(centre_file, components, fmt='%.6f')


