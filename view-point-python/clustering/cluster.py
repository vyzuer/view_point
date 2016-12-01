import kmeans
import ward
import sys
import dbscan
import ap
import numpy as np

# dbPath = "/home/scps/myDrive/Copy/Flickr-code/DBR/clustering_1004/"
# file_name = "/home/scps/myDrive/Copy/Flickr-code/DBR/clustering_1004/segments.list"


def cluster_k(db_path, file_name, file_name_p, n_clusters):
    score = kmeans.k_means(dump_path=db_path, file_name=file_name, file_name_p=file_name_p, n_clusters=int(n_clusters))    
    print '{0}:{1}'.format(n_clusters, score)

def cluster_w(db_path, file_name, n_clusters):
    score = ward.cluster(dump_path=db_path, file_name=file_name, n_clusters=int(n_clusters))    
    print '{0}:{1}'.format(n_clusters, score)

def cluster_dbscan(db_path, file_name):
    dbscan.cluster(db_path, file_name)

def cluster_ap(db_path, file_name):
    ap.cluster(db_path, file_name)

def compute_popularity(db_path):
    # input
    f_saliency = db_path + "km_saliency.list"
    f_labels = db_path + "km_labels.list"
    f_num_clusters = db_path + "_num_clusters.info"

    # output
    f_popularity = db_path + "popularity.score"

    sal_list = np.loadtxt(f_saliency)
    labels_list = np.loadtxt(f_labels, dtype=np.int)
    num_clusters = np.loadtxt(f_num_clusters, dtype=np.int)

    cluster_bins = np.zeros(num_clusters)
    cluster_saliency = np.zeros(num_clusters)
    cluster_popularity = np.zeros(num_clusters)

    for i in range(len(sal_list)):
        label = labels_list[i]
        saliency = sal_list[i]
        
        cluster_bins[label] += 1
        cluster_saliency[label] += saliency

    max_N = np.amax(cluster_bins)

    for i in range(num_clusters):
        if cluster_bins[i] > 0:
            cluster_popularity[i] = (cluster_bins[i]/max_N + cluster_saliency[i]/cluster_bins[i])/2

    np.savetxt(f_popularity, cluster_popularity, fmt='%.6f')

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print "Usage : test.py dump_path feature.list num_clusters"
        sys.exit(0)
    
    db_path = sys.argv[1]
    file_name = sys.argv[2]
    file_name_p = sys.argv[3]
    n_clusters = sys.argv[4]
    # cluster_ap(db_path, file_name)
    # cluster_dbscan(db_path, file_name)
    cluster_k(db_path, file_name, file_name_p, n_clusters)    
    # cluster_w(db_path, file_name, n_clusters)    

    compute_popularity(db_path)

