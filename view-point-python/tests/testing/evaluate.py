import sys
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, accuracy_score

def remove_average_photographs(data1, data2):
    x = data1.size

    s1 = np.around(data1, decimals=0)
    s2 = np.around(data2, decimals=0)

    X = []
    Y = []

    good = 0.75
    bad = 0.25

    for i in range(x):
        if data1[i] < bad or data2[i] < bad:
            X.append(s1[i])
            Y.append(s2[i])

        elif data1[i] > good or data2[i] > good:
            X.append(s1[i])
            Y.append(s2[i])

    Y = np.asarray(Y)
    X = np.asarray(X)

    return X, Y

def process(dump_path, dump_path2, file_name):
    scores1_file = dump_path2 + 'aesthetic.scores'
    scores2_file = dump_path2 + file_name
    # scores1_file = dump_path2 + file_name

    s1 = np.loadtxt(scores1_file)
    s2 = np.loadtxt(scores2_file)

    out_file = dump_path + 'correlation.dump'

    mse = mean_squared_error(s2, s1)

    # s1 = np.around(s1, decimals=0)
    # s2 = np.around(s2, decimals=0)

    s1, s2 = remove_average_photographs(s1, s2)

    r = recall_score(s1, s2)
    p = precision_score(s1, s2)
    a = accuracy_score(s1, s2)
    # r1 = pearsonr(s1, s2)
    # r2 = spearmanr(s1, s2)

    print a, p, r, mse

    np.savetxt(out_file, [a, p, r, mse], fmt='%f')


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "Usage : dump_path1 dump_path2 file_name"
        sys.exit(0)

    dump_path1 = sys.argv[1]
    dump_path2 = sys.argv[2]
    file_name = sys.argv[3]
    
    process(dump_path1, dump_path2, file_name)

