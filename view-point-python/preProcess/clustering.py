import Image
import time
from matplotlib.patches import Ellipse

import numpy as np
import pylab as pl

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import scale

class clustering:
    def __init__(self, db_path, file_name, n_clusters=10):
        self.db_path = db_path
        self.file_name = file_name

        self.labels = self.__cluster(db_path, file_name, n_clusters)

    def __cluster(self, db_path, file_name, n_clusters):
        # Obtain data from file.
        #feature_file = 'feature.list'
        feature_file = db_path + file_name
        data = np.loadtxt(feature_file, unpack=True)
        m1 = data[1]
        
        X = np.transpose(data)
        X = scale(X)
        labels_true = np.zeros(len(m1))
        print labels_true.shape
        
        # Compute clustering with Means
        k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=10000)

        t0 = time.time()
        k_means.fit(X)
        t_batch = time.time() - t0
        k_means_labels = k_means.labels_
        #print k_means_labels
        label_file = db_path + "labels.list"
        fp = open(label_file, 'w')
        for i in k_means_labels:
            fp.write("%d\n" % i)
        fp.close()

        num_cluster_file = db_path + "_num_clusters.info"
        fp = open(num_cluster_file, 'w')
        fp.write("%d" % n_clusters)
        fp.close()

        k_means_cluster_centers = k_means.cluster_centers_
        k_means_labels_unique = np.unique(k_means_labels)
        print metrics.silhouette_score(X, k_means_labels, metric='euclidean')

        return k_means_labels
