import itertools
import sys, os
import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.decomposition import RandomizedPCA
from sklearn import mixture
from sklearn.externals import joblib
from sklearn import preprocessing

def gmm(X, s_path, ext):
    
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X)

    clf = mixture.GMM(n_components=10, covariance_type='full')
    # clf = mixture.GMM(n_components=10, covariance_type='full', thresh=0.00001, n_iter=1000)
    clf.fit(X)
    
    xmin, xmax = np.amin(X[:,1]), np.amax(X[:,1])
    ymin, ymax = np.amin(X[:,0]), np.amax(X[:,0])
    print xmin, ymin, xmax, ymax

    # print clf.predict(clf.means_)

    if 0:
        # Plot the winner
        color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
        Y_ = clf.predict(X)
        for i, (mean, covar, color) in enumerate(zip(clf.means_, clf.covars_,
                                                     color_iter)):
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        
        
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xticks(())
        plt.yticks(())
        plt.title('Selected GMM')
        plt.subplots_adjust(hspace=.35, bottom=.02)
        plt.show()

    x, y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    positions = np.vstack([y.ravel(), x.ravel()])
    prob_score, response = clf.score_samples(zip(*positions))
    f = np.exp(prob_score)

    # scale results for later user
    min_max_scaler = preprocessing.MinMaxScaler()
    # f = min_max_scaler.fit_transform([f])
    f = np.reshape(f, x.shape)
    
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(np.rot90(f), cmap=plt.cm.ocean_r, extent=[xmin, xmax, ymin, ymax])
    plt.colorbar()

    # dump plot
    dump_dir = s_path + '/' + ext 
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    plot_path = dump_dir + "/gmm.png"
    plt.savefig(plot_path, dpi=400)
    plt.close()

    # dump the gmm model
    model_dump = dump_dir + "/model/"
    if not os.path.exists(model_dump):
        os.makedirs(model_dump)

    model_path = model_dump + "/gmm.pkl"

    scaler_dump = dump_dir + "/scaler/"
    if not os.path.exists(scaler_dump):
        os.makedirs(scaler_dump)

    scaler_path = scaler_dump + "/scaler.pkl"
    mm_scaler_path = scaler_dump + "/mm_scaler.pkl"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(min_max_scaler, mm_scaler_path)

def gmm_context(X, s_path, ext):
    
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X)

    n_samples, n_dims = X.shape

    # Fit a mixture of Gaussians with EM
    gmm = mixture.GMM(n_components=10, covariance_type='full')
    # gmm = mixture.GMM(n_components=10, covariance_type='full', thresh=0.00001, n_iter=1000)
    gmm.fit(X)

    clf = gmm
    
    xmin, xmax = np.amin(X[:,1]), np.amax(X[:,1])
    ymin, ymax = np.amin(X[:,0]), np.amax(X[:,0])
    # print xmin, ymin, xmax, ymax

    dump_dir = s_path + '/' + ext 
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    prob_score, response = clf.score_samples(X)
    f = np.exp(prob_score)

    # scale results for later user
    min_max_scaler = preprocessing.MinMaxScaler()
    f = min_max_scaler.fit_transform([f])

    # dump the gmm model
    model_dump = dump_dir + "/model/"
    if not os.path.exists(model_dump):
        os.makedirs(model_dump)

    model_path = model_dump + "/gmm.pkl"

    scaler_dump = dump_dir + "/scaler/"
    if not os.path.exists(scaler_dump):
        os.makedirs(scaler_dump)

    scaler_path = scaler_dump + "/scaler.pkl"
    mm_scaler_path = scaler_dump + "/mm_scaler.pkl"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(min_max_scaler, mm_scaler_path)


def find_points(data, t_info=None):

    fv = data
    if t_info is not None:
        fv = np.concatenate([fv, np.reshape(t_info, (-1, 1))], axis=1)

    return fv


def get_time_info(f_time_list):
    time_info = np.loadtxt(f_time_list, dtype='string')
    # print time_info.size
    time_list = np.zeros(time_info.size)

    i = 0
    for t_info in time_info:
        # print t_info
        t_array = t_info.split(':')
        # t_hrs = int(t_array[0]) + int(t_array[1])/60.0 + float(t_array[2])/3600.0
        t_hrs = int(t_array[0]) + int(t_array[1])/60.0
        # print t_hrs
        time_list[i] = t_hrs
        i += 1

    return time_list
        

def process_context(dataset_path, dump_path, gmm_3d=False):

    gmm_model_path = dump_path + "/gmm_models/"

    f_pos_list = dataset_path + "/geo.info"
    f_time_list = dataset_path + "/time.info"

    assert os.path.isfile(f_pos_list)
    assert os.path.isfile(f_time_list)

    data = np.loadtxt(f_pos_list)

    n_samples = data.shape[0]

    if n_samples > 10 :

        if gmm_3d == True:
            time_list = get_time_info(f_time_list)
            X = find_points(data, time_list)
        else:
            X = data

        print data.shape, X.shape
    
        if gmm_3d == False:
            gmm(X, gmm_model_path, 'basic')
        else:
            gmm_context(X, gmm_model_path, 'time')

