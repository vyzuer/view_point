import sys
import os
import time
import numpy as np

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.externals import joblib
from sklearn.mixture import VBGMM
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, normalize

def cluster_vbgmm(X):
    print "started vbgmm clustering..."
    model = VBGMM(covariance_type='full')
    model.fit(X)
    print "number of components = %d" % model.n_components
    print "done."

    return model 

def cluster_ms(X):
    print "started meanshift clustering..."
    bandwidth = 4.0
    model = MeanShift(bandwidth=bandwidth)
    model.fit(X)
    print "done."

    return model 

def cluster_k(X, n_clusters):
    print "started k-means..."
    # k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=10000000, tol=0.0000001, n_jobs=-1)
    # k_means = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=1, max_iter=10, tol=0.1, max_no_improvement=None)
    k_means = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10, max_iter=5000, tol=0.0, max_no_improvement=200)

    k_means.fit(X)
    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, k_means.labels_))
    print "k-means done."

    return k_means

def cluster_w(db_path, file_name, n_clusters):
    score = ward.cluster(dump_path=db_path, file_name=file_name, n_clusters=int(n_clusters))    
    print '{0}:{1}'.format(n_clusters, score)

def cluster_dbscan(X):
    print "started DBSCAN clustering..."
    X = StandardScaler().fit_transform(X)

    eps = 51.0
    model = DBSCAN(eps=eps, min_samples=4).fit(X)

    labels = model.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print "Number of clusters = %d" % n_clusters_
    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

    print "done."

    return model, n_clusters_

def scale_data_0(X):
    frame_scale     = 1.0
    shape_scale     = 1.0
    surf_scale      = 1.0
    hog_scale       = 1.0
    rgb_scale       = 1.0

    X[:,0:2] = frame_scale*X[:,0:2]
    X[:,2:102] = shape_scale*X[:,2:102]
    X[:,102:166] = surf_scale*X[:,102:166]
    X[:,166:230] = hog_scale*X[:,166:230]
    X[:,230:] = rgb_scale*X[:,230:]

    return X

def scale_data(X):
    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    X = normalize(X)

    return X

def cluster_ap(X):
    print "started AP clustering..."
    print X.shape

    testing = False
    # testing = True
    if testing == True:
        for i in range(50):
            damping = 0.95 + i*0.001
            print damping
            model = AffinityPropagation(damping=damping, max_iter=5000, convergence_iter=100, verbose=True)
            model.fit(X)

            n_clusters_ = len(model.cluster_centers_indices_)
            print "Number of clusters = %d" % n_clusters_
            print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, model.labels_))

    damping = 0.5
    model = AffinityPropagation(damping=damping, max_iter=5000, convergence_iter=100, verbose=True)
    model.fit(X)

    n_clusters_ = len(model.cluster_centers_indices_)
    print "Number of clusters = %d" % n_clusters_
    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, model.labels_))

    print "done."

    return model, n_clusters_

def process(dump_path, n_clusters=4):
    seg_path = dump_path + '/segments/'
    data_file = seg_path + '/segments.list'

    ap_c = False
    ap_c = True

    data = np.loadtxt(data_file)
    # data = scale_data(data)
    # data = data[:, :]
    print data.shape

    timer = time.time()
    if ap_c == True:
        model, n_clusters = cluster_ap(data)
    else:
        model = cluster_k(data, n_clusters)    

    print "Clustering time : ", time.time() - timer

    # model, n_clusters = cluster_dbscan(data)
    # model = cluster_ms(data)    
    # model = cluster_vbgmm(data)    
    # cluster_w(db_path, file_name, n_clusters)    

    labels = model.labels_
    # labels = model.predict(data)

    model_path = dump_path + '/cluster_model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_file = model_path + 'cluster.pkl'
    joblib.dump(model, model_file)

    label_file = seg_path + '/labels.list'
    np.savetxt(label_file, labels, fmt='%d')

    if ap_c == True:
        centers_file = seg_path + '/centers.list'
        np.savetxt(centers_file, model.cluster_centers_indices_, fmt='%d')

    cluster_count = np.bincount(labels)
    f_cluster_count = seg_path + '/cluster.count'
    np.savetxt(f_cluster_count, cluster_count, fmt='%d')

    num_cluster_file = seg_path + "_num_clusters.info"
    fp = open(num_cluster_file, 'w')
    fp.write("%d" % n_clusters)
    fp.close()

    # testing
    labels = model.predict(data)
    label_file = seg_path + '/labels_test.list'
    np.savetxt(label_file, labels, fmt='%d')
 
