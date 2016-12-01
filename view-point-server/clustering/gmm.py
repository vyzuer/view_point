import itertools
import sys, os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from sklearn.externals import joblib

def gmm(X, s_path):
    
    x_scale = 4
    y_scale = 3
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 20)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    # cv_types = ['full']
    X[:,0] = [y_scale - x*y_scale for x in X[:,0]]
    X[:,1] = [x*x_scale for x in X[:,1]]

    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    
    bic = np.array(bic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    bars = []
    
    # Plot the BIC scores
    spl = plt.subplot(3, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    
    # Plot the winner
    splot = plt.subplot(3, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_,
                                                 color_iter)):
        plt.scatter(X[Y_ == i, 1], X[Y_ == i, 0], .8, color=color)
    
    plt.xlim(0, x_scale)
    plt.ylim(0, y_scale)
    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model, 2 components')
    # plt.subplots_adjust(hspace=.35, bottom=.02)

    # Plot the results:
    splot3 = plt.subplot(3, 1, 3)

    xmin, xmax = 0, x_scale
    ymin, ymax = 0, y_scale
    x, y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([y.ravel(), x.ravel()])
    prob_score, response = clf.score_samples(zip(*positions))
    print prob_score.shape
    f = np.exp(prob_score)
    f = np.reshape(f, x.shape)
    
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(np.rot90(f), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    # plt.plot(X[:,1], X[:,0], 'k.', markersize=0.1)
    plt.colorbar()

    # dump plot
    dump_dir = s_path + "/gmm_dumps/"
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    plot_path = dump_dir + "gmm.png"
    plt.savefig(plot_path)
    plt.close()

    # dump the gmm model
    model_path = dump_dir + "gmm.pkl"
    joblib.dump(clf, model_path)


def find_points(data, a_score):    
    X = []

    num_points = len(a_score)
    for i in range(num_points):
        count = int(a_score[i]*10)
        for j in range(count):
            X.append(data[i])
    
    X = np.array(X)
    print X.shape
    return X

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage : gmm.py dump_path"
        sys.exit(0)

    dump_path = sys.argv[1]

    f_num_clusters = dump_path + "_num_clusters.info"
    clusters_info = dump_path + "SegClustersInfo/"

    num_clusters = np.loadtxt(f_num_clusters, dtype=np.int)

    for i in range(num_clusters):
        s_path = clusters_info + str(i)
        f_pos_list = s_path + "/pos.list"

        if os.path.isfile(f_pos_list):
            f_a_score = s_path + "/aesthetic.scores"

            a_score = np.loadtxt(f_a_score)
            data = np.loadtxt(f_pos_list)
            n_samples = data.size

            if n_samples < 40:
                continue

            X = find_points(data, a_score)

            gmm(X, s_path)


