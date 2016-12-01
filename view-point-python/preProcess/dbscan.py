import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import scale

data_file = "feature.list"
# Obtain data from file.
data = np.loadtxt(data_file, unpack=True)
print data.shape
m1 = data[1]

# Compute DBSCAN
X = np.transpose(data)
X = scale(X)
np.savetxt("scaled.fv", X)    
labels_true = np.zeros(len(m1))
print labels_true.shape

db = DBSCAN(eps=20, min_samples=10).fit(X)
core_samples = db.core_sample_indices_
labels = db.labels_
print labels

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

