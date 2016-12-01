import time as time
import numpy as np
from sklearn.cluster import Ward
from sklearn.preprocessing import scale


def cluster(dump_path, file_name, n_clusters=200):
    # Obtain data from file.
    #feature_file = 'feature.list'
    data = np.loadtxt(file_name, unpack=True)
    m1 = data[1]
    
    X = np.transpose(data)
    X = scale(X)
    labels_true = np.zeros(len(m1))
    
    ###############################################################################
    # Compute clustering
    print("Compute unstructured hierarchical clustering...")
    st = time.time()
    ward = Ward(n_clusters=n_clusters).fit(X)
    label = ward.labels_
    print("Elapsed time: ", time.time() - st)
    print("Number of points: ", label.size)

    label_file = dump_path + "ward_labels.list"
    fp = open(label_file, 'w')
    for i in label:
        fp.write("%d\n" % i)
    fp.close()

    num_cluster_file = dump_path + "_num_clusters_ward.info"
    fp = open(num_cluster_file, 'w')
    fp.write("%d" % n_clusters)
    fp.close()


    cluster_centers = ward.cluster_centers_
    
    score = 0.0
    # print "evaluating performance..."
    # score = metrics.silhouette_score(X, label, metric='euclidean', sample_size=20000)
    # print "evaluation done."
    # score = metrics.silhouette_samples(X, k_means_labels, metric='euclidean', sample_size=1000)
    # score = np.sum(score)/len(score)

    return score

