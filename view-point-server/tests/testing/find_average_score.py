import sys
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, accuracy_score


def process(file_name):
    x = np.loadtxt(file_name)
    print np.mean(x, axis=0)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Usage : file_name"
        sys.exit(0)

    file_name = sys.argv[1]
    
    process(file_name)

