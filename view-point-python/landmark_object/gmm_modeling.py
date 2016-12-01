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

def gmm(X, s_path, ext, data=None):
    
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X)
    X_0 = scaler.transform(data)

    x_scale = 1
    y_scale = 1
    lowest_bic = np.infty
    bic = []

    n_samples, n_dims = X.shape

    max_components = min(n_samples, 100)

    n_components_range = range(6, max_components, 10)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['full']

    best_gmm = None

    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, tol=0.00001, n_iter=1000)
            gmm.fit(X)
            bic.append(np.abs(gmm.bic(X)))
            
            if np.abs(bic[-1]) < lowest_bic:
                lowest_bic = np.abs(bic[-1])
                best_gmm = gmm
    
    print s_path, best_gmm.n_components

    # local search for best fit around the best global
    lowest_bic = np.infty
    bic = []
    n_components_range = range(best_gmm.n_components-5, best_gmm.n_components+5)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['full']

    best_gmm = None

    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, tol=0.00001, n_iter=1000)
            gmm.fit(X)
            bic.append(np.abs(gmm.bic(X)))
            
            if np.abs(bic[-1]) < lowest_bic:
                lowest_bic = np.abs(bic[-1])
                best_gmm = gmm
    
    bic = np.array(bic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    print s_path, best_gmm.n_components
    bars = []
    
    # Plot the BIC scores
    spl = plt.subplot(4, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    # plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)+3) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    
    xmin, xmax = np.amin(X[:,1]), np.amax(X[:,1])
    ymin, ymax = np.amin(X[:,0]), np.amax(X[:,0])
    # print xmin, ymin, xmax, ymax

    # Plot the winner
    splot = plt.subplot(4, 1, 2)
    plt.axis('equal')
    Y_ = clf.predict(X)
    for i, (mean, color) in enumerate(zip(clf.means_,
                                                 color_iter)):
        plt.scatter(X[Y_ == i, 1], X[Y_ == i, 0], .8, color=color)
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(())
    plt.yticks(())
    # plt.title('Selected GMM: full model')
    # plt.subplots_adjust(hspace=.35, bottom=0.2)

    splot3 = plt.subplot(4, 1, 3)
    plt.xticks(())
    plt.yticks(())
    plt.axis('equal')
    plt.scatter(X_0[:,1], X_0[:,0], 0.6)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    # Plot the results:
    splot4 = plt.subplot(4, 1, 4)
    plt.axis('equal')

    x, y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    positions = np.vstack([y.ravel(), x.ravel()])
    prob_score, response = clf.score_samples(zip(*positions))
    f = np.exp(prob_score)

    # scale results for later user
    min_max_scaler = preprocessing.MinMaxScaler()
    f = min_max_scaler.fit_transform(f)
    f = np.reshape(f, x.shape)
    
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(np.rot90(f), cmap=plt.cm.ocean_r, extent=[xmin, xmax, ymin, ymax])
    # plt.plot(X[:,1], X[:,0], 'k.', markersize=0.1)
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

def gmm_context(X, s_path, ext, data=None):
    
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X)
    # X_0 = scaler.transform(data)

    x_scale = 1
    y_scale = 1
    lowest_bic = np.infty
    bic = []

    n_samples, n_dims = X.shape

    max_components = min(n_samples, 100)

    n_components_range = range(6, max_components, 10)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['full']

    best_gmm = None

    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, tol=0.00001, n_iter=1000)
            gmm.fit(X)
            bic.append(np.abs(gmm.bic(X)))
            
            if np.abs(bic[-1]) < lowest_bic:
                lowest_bic = np.abs(bic[-1])
                best_gmm = gmm
    
    print s_path, best_gmm.n_components

    # local search for best fit around the best global
    lowest_bic = np.infty
    bic = []
    n_components_range = range(best_gmm.n_components-5, best_gmm.n_components+5)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['full']

    best_gmm = None

    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type, tol=0.00001, n_iter=1000)
            gmm.fit(X)
            bic.append(np.abs(gmm.bic(X)))
            
            if np.abs(bic[-1]) < lowest_bic:
                lowest_bic = np.abs(bic[-1])
                best_gmm = gmm
    
    bic = np.array(bic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    print s_path, best_gmm.n_components
    bars = []
    
    # Plot the BIC scores
    spl = plt.subplot(3, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    # plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)+3) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    
    xmin, xmax = np.amin(X[:,1]), np.amax(X[:,1])
    ymin, ymax = np.amin(X[:,0]), np.amax(X[:,0])
    # print xmin, ymin, xmax, ymax

    # Plot the winner
    splot = plt.subplot(3, 1, 2)
    plt.axis('equal')
    Y_ = clf.predict(X)
    for i, (mean, color) in enumerate(zip(clf.means_,
                                                 color_iter)):
        plt.scatter(X[Y_ == i, 1], X[Y_ == i, 0], .8, color=color)
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks(())
    plt.yticks(())
    # plt.title('Selected GMM: full model')
    # plt.subplots_adjust(hspace=.35, bottom=0.2)

    splot3 = plt.subplot(3, 1, 3)
    plt.xticks(())
    plt.yticks(())
    plt.axis('equal')
    plt.scatter(data[:,1], data[:,0], 0.6)
    x_min = data.min(axis=0)[1]
    y_min = data.min(axis=0)[0]
    x_max = data.max(axis=0)[1]
    y_max = data.max(axis=0)[0]
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    dump_dir = s_path + '/' + ext 
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    plot_path = dump_dir + "/gmm.png"
    plt.savefig(plot_path, dpi=400)
    plt.close()

    prob_score, response = clf.score_samples(X)
    f = np.exp(prob_score)

    # scale results for later user
    min_max_scaler = preprocessing.MinMaxScaler()
    f = min_max_scaler.fit_transform(f)

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

def find_points_0(data, a_score):    
    X = []

    num_points = len(a_score)
    for i in range(num_points):
        if a_score[i] > 0.02:
            X.append(data[i])
    
    X = np.array(X)
    return X

def find_points_1(data, a_score, t_info=None, w_info=None):
    X = []

    num_points = a_score.size

    fv = data
    if t_info is not None:
        fv = np.concatenate([fv, np.reshape(t_info, (-1, 1))], axis=1)
    if w_info is not None:
        fv = np.concatenate([fv, w_info], axis=1)

    for i in range(num_points):
        if a_score[i] > 0.1:
            X.append(fv[i])
            count = int(np.round(a_score[i]*10))
            for j in range(count):
                dist = (2*np.random.random((1,2))-1)/50000
                # print('%.8f %.8f' % (dist[0][0], dist[0][1]))
                new_point = data[i] + dist[0]

                fv[i][0:2] = new_point

                X.append(fv[i])
        else:
            X.append(fv[i])
    
    X = np.array(X)
    return X

def find_points(data, a_score, t_info=None, w_info=None):

    fv = data
    if t_info is not None:
        fv = np.concatenate([fv, np.reshape(t_info, (-1, 1))], axis=1)
    if w_info is not None:
        fv = np.concatenate([fv, w_info], axis=1)

    return fv

def __process(model_path, dump_path, ext):

    f_num_clusters = model_path + "/segments/_num_clusters.info"
    gmm_model_path = model_path + "/gmm_models/"
    clusters_info = dump_path + "/lm_objects/"

    num_clusters = np.loadtxt(f_num_clusters, dtype=np.int)

    # find the 

    for i in range(num_clusters):
    # for i in range(1):
        s_path = clusters_info + str(i)
        s_gmm_model = gmm_model_path + str(i)
        f_pos_list = s_path + "/geo.info"

        if os.path.isfile(f_pos_list):
            f_a_score = s_path + "/aesthetic.scores"

            a_score = np.loadtxt(f_a_score)
            data = np.loadtxt(f_pos_list)
            n_samples = a_score.size

            if n_samples > 10 :
    
                X = find_points(data, a_score)
    
                print data.shape, X.shape
                n_samples = X.size
    
                if n_samples > 200:
                    gmm(X, s_gmm_model, ext, data)


def get_time_info(f_time_list):
    time_info = np.loadtxt(f_time_list, dtype='string')
    print time_info.size
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
        
def find_face_points_0(face_info, data, a_score, t_info=None, w_info=None):
    X = []

    num_points = a_score.size

    fv = data
    if t_info is not None:
        fv = np.concatenate([fv, np.reshape(t_info, (-1, 1))], axis=1)
    if w_info is not None:
        fv = np.concatenate([fv, np.reshape(face_info, (-1, 1)), w_info], axis=1)

    for i in range(num_points):
        if face_info[i] > 0:
            if a_score[i] > 0.1:
                X.append(fv[i])
                count = int(np.round(a_score[i]*10))
                for j in range(count):
                    dist = (2*np.random.random((1,2))-1)/50000
                    # print('%.8f %.8f' % (dist[0][0], dist[0][1]))
                    new_point = data[i] + dist[0]

                    fv[i][0:2] = new_point

                    X.append(fv[i])
            else:
                X.append(fv[i])
    
    X = np.array(X)
    return X

def find_face_points(face_info, data, a_score, t_info=None, w_info=None):

    fv = data
    if t_info is not None:
        fv = np.concatenate([fv, np.reshape(t_info, (-1, 1))], axis=1)
    if w_info is not None:
        fv = np.concatenate([fv, np.reshape(face_info, (-1, 1)), w_info], axis=1)

    return fv[face_info != 0]

def get_reduced_data(w_info, model_path):
    pca_dump = model_path + '/w_pca_model/pca.pkl'
    scaler_dump = model_path + '/w_pca_model/scaler.pkl'

    pca = joblib.load(pca_dump)
    scaler = joblib.load(scaler_dump)

    w_info = scipy.delete(w_info, 7, 1)[:, :-3]
    w_info = scaler.transform(w_info)
    w_info = pca.transform(w_info)

    return w_info


def process_human_object(model_path, dump_path, ext, model_type='basic'):

    s_gmm_model = model_path + "/gmm_models/human_obj/"

    f_face_list = dump_path + "/face.list"
    f_pos_list = dump_path + "/geo.info"
    f_time_list = dump_path + "/time.info"
    f_a_score = dump_path + "/aesthetic.scores"
    f_weather_info = dump_path + '/weather.info'

    assert os.path.isfile(f_pos_list)
    assert os.path.isfile(f_pos_list)
    assert os.path.isfile(f_time_list)
    assert os.path.isfile(f_a_score)
    assert os.path.isfile(f_weather_info)

    a_score = np.loadtxt(f_a_score)
    data = np.loadtxt(f_pos_list)
    face_info = np.loadtxt(f_face_list)

    if model_type == 'basic':
        X = find_face_points(face_info, data, a_score)
    elif model_type == 'time':
        time_info = get_time_info(f_time_list)
        X = find_face_points(face_info, data, a_score, time_info)
    else:
        time_info = get_time_info(f_time_list)
        weather_info = np.loadtxt(f_weather_info)
        # remove extra details
        weather_info = get_reduced_data(weather_info, model_path)
        X = find_face_points(face_info, data, a_score, time_info, weather_info)
    
    print data.shape, X.shape
    
    if model_type == 'basic':
        gmm(X, s_gmm_model, ext, data)
    else:
        gmm_context(X, s_gmm_model, ext, data)


def reduce_dimension(model_path, dump_path):
    f_weather_info = dump_path + '/weather.info'
    weather_info = np.loadtxt(f_weather_info)
    
    w_info = scipy.delete(weather_info, 7, 1)[:, :-3]
    scaler = preprocessing.StandardScaler()

    scaled_info = scaler.fit_transform(w_info)
    
    pca = RandomizedPCA(n_components=4)
    pca.fit(scaled_info)

    # dump pca and scaler
    dump_dir = model_path + '/w_pca_model/'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    pca_dump = dump_dir + 'pca.pkl'
    scaler_dump = dump_dir + 'scaler.pkl'

    joblib.dump(pca, pca_dump)
    joblib.dump(scaler, scaler_dump)

    return pca, scaler

def process_context(model_path, dump_path, ext, model_type='basic'):

    f_num_clusters = model_path + "/segments/_num_clusters.info"
    gmm_model_path = model_path + "/gmm_models/"
    clusters_info = dump_path + "/lm_objects/"

    num_clusters = np.loadtxt(f_num_clusters, dtype=np.int)

    # reduce weather dimensionality
    pca, w_scaler = None, None
    if model_type == 'weather':
        pca, w_scaler = reduce_dimension(model_path, dump_path)

    for i in range(num_clusters):
    # for i in range(0):
        s_path = clusters_info + str(i)
        s_gmm_model = gmm_model_path + str(i)

        f_pos_list = s_path + "/geo.info"
        f_time_list = s_path + "/time.info"
        f_a_score = s_path + "/aesthetic.scores"
        f_weather_info = s_path + '/weather.info'

        assert os.path.isfile(f_pos_list)
        assert os.path.isfile(f_time_list)
        assert os.path.isfile(f_a_score)
        assert os.path.isfile(f_weather_info)

        a_score = np.loadtxt(f_a_score)
        data = np.loadtxt(f_pos_list)

        n_samples = a_score.size

        if n_samples > 10 :

            if model_type == 'basic':
                X = find_points(data, a_score)
            elif model_type == 'time':
                time_info = get_time_info(f_time_list)
                X = find_points(data, a_score, time_info)
            else:
                time_info = get_time_info(f_time_list)
                weather_info = np.loadtxt(f_weather_info)
                # remove extra details
                weather_info = scipy.delete(weather_info, 7, 1)[:, :-3]
                weather_info = w_scaler.transform(weather_info)
                weather_info = pca.transform(weather_info)
                X = find_points(data, a_score, time_info, weather_info)
                
    
            print data.shape, X.shape
    
            if model_type == 'basic':
                gmm(X, s_gmm_model, ext, data)
            else:
                gmm_context(X, s_gmm_model, ext, data)

