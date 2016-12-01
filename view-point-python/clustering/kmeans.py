import Image
import time
import os
import numpy as np
import pylab as pl

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
from sklearn.preprocessing import scale, normalize
from sklearn.externals import joblib

def k_means(dump_path, file_name, file_name_p, n_clusters=10):
    # Obtain data from file.
    #feature_file = 'feature.list'
    data = np.loadtxt(file_name)
    n_samples, n_dim = data.shape
    print data.shape
    
    # surf
    # X = data[:, 0:64]
    # hog
    # X = data[:, 64:96]
    # use only RGB
    # X = data[:, 96:864]
    X = data
    # X = scale(data)
    # X = normalize(X, norm='l2')
    
    # Compute clustering with Means
    print "started k-means..."
    # k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=10000000, tol=0.0000001, n_jobs=-1)
    # k_means = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=1, max_iter=10, tol=0.1, max_no_improvement=None)
    k_means = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=2000, tol=0.0, max_no_improvement=50)

    k_means.fit(X)
    print "k-means done."
    k_means_labels = k_means.labels_
    k_means_inertia = k_means.inertia_
    #print k_means_labels
    label_file = dump_path + "km_labels.list"
    np.savetxt(label_file, k_means_labels, fmt='%d')

    num_cluster_file = dump_path + "_num_clusters.info"
    fp = open(num_cluster_file, 'w')
    fp.write("%d" % n_clusters)
    fp.close()

    inertia_file = dump_path + "_inertia.info"
    fp = open(inertia_file, 'w')
    inertia_value = k_means_inertia/n_samples
    fp.write("%f" % inertia_value)
    fp.close()

    k_means_cluster_centers = k_means.cluster_centers_
    
    centre_file = dump_path + "_centers.info"
    np.savetxt(centre_file, k_means_cluster_centers, fmt='%.10f')

    # dump model
    model_path = dump_path + "cluster_model/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_file = model_path + "cluster.pkl"
    joblib.dump(k_means, model_file)

    data = np.loadtxt(file_name_p)
    k_means_labels = k_means.predict(data)
    label_file = dump_path + "labels.list"
    np.savetxt(label_file, k_means_labels, fmt='%d')

    score = 0

    # print "evaluating performance..."
    # score = metrics.silhouette_score(X, k_means_labels, metric='euclidean', sample_size=20000)
    # print "evaluation done."
    # score = metrics.silhouette_samples(X, k_means_labels, metric='euclidean', sample_size=1000)
    # score = np.sum(score)/len(score)

    return score

