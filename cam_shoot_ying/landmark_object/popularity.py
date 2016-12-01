import sys
import time
import os
import glob
import numpy as np
import shutil
from sklearn.externals import joblib

import landmark_object.gmm_modeling as gmm_mod

# load globals only once and share
gmm_model = None
scaler_model = None
a_scores = None
time_list = None
geo_list = None
owner_list = None
favs_list = None
views_list = None
year_list = None

def reset_global_vars():
    global gmm_model
    global scaler_model
    global a_scores
    global time_list
    global geo_list
    global owner_list
    global favs_list
    global views_list
    global year_list

    gmm_model = None
    scaler_model = None
    a_scores = None
    geo_list = None
    owner_list = None
    favs_list = None
    views_list = None
    year_list = None


def set_global_vars(dataset_path, dump_path, ext):

    global gmm_model
    global scaler_model
    global a_scores
    global time_list
    global geo_list
    global owner_list
    global favs_list
    global views_list
    global year_list


    # load gmm models
    model_path = dump_path + "/gmm_models/"
    model_base_path = model_path + "/" + ext
    gmm_model_path = model_base_path + "/model/gmm.pkl"
    scaler_model_path = model_base_path + "/scaler/scaler.pkl"

    assert os.path.exists(gmm_model_path)

    gmm_model = joblib.load(gmm_model_path)
    scaler_model = joblib.load(scaler_model_path)

    a_score_list = dump_path + 'ying_ascore.list'
    a_scores = np.loadtxt(a_score_list)

    f_pos_list = dataset_path + "/geo.info"
    f_time_list = dataset_path + "/time.info"
    f_owner_list = dataset_path + "/owners.list"
    f_views_list = dataset_path + "/view.list"
    f_favs_list = dataset_path + "/favs.list"
    f_year_list = dataset_path + "/year.list"

    assert os.path.isfile(f_pos_list)
    assert os.path.isfile(f_time_list)
    assert os.path.isfile(f_owner_list)
    assert os.path.isfile(f_views_list)
    assert os.path.isfile(f_favs_list)
    assert os.path.isfile(f_year_list)

    geo_list = np.loadtxt(f_pos_list)
    time_list = gmm_mod.get_time_info(f_time_list)
    owner_list = np.loadtxt(f_owner_list, dtype='int')
    views_list = np.loadtxt(f_views_list, dtype='int', skiprows=1)
    favs_list = np.loadtxt(f_favs_list, dtype='int', skiprows=1)
    year_list = np.loadtxt(f_year_list, dtype='int', skiprows=1)


def compute_soc_att(ids):
    v_list = views_list[ids]
    f_list = favs_list[ids]

    w1, w2 = 1.0, 1.0

    sa = np.sum(w1*v_list + w2*f_list)

    return sa

def compute_temporal_density(ids):

    t_list = year_list[ids]

    t_density = 0.0
    diff = np.amax(t_list) - np.amin(t_list)
    if diff > 0:
        t_density = t_list.size/diff

    return t_density

def compute_density(ids, i):

    N = geo_list[ids].size

    v, w = np.linalg.eigh(gmm_model._get_covars()[i])

    density = N/np.prod(v)

    return density

def compute_apopcon(ids):
    t_list = year_list[ids]
    t0 = np.mean(t_list)

    apopcon = np.sqrt(np.sum(np.square(t_list-t0)))

    return apopcon

def compute_pop(ids):
    o_list = owner_list[ids]
    # print 'total owners : ', o_list.size

    uowners, cnt = np.unique(o_list, return_counts=True)
    num_owners = len(uowners)

    # print 'num_owners : ', num_owners

    a_pop = num_owners + np.sum(np.log(cnt))

    return a_pop

def compute_pquality(ids):
    scores_list = a_scores[ids]
    # print 'total images : ', scores_list.size

    pq = np.mean(scores_list)

    return pq

def compute_vp_quality(ids, i):
    quality = 0.0

    # photo quality
    p_quality = compute_pquality(ids)

    # area popularity
    a_pop = compute_pop(ids)
    # print a_pop

    a_pop_con = compute_apopcon(ids)
    # print a_pop_con

    sp_den = compute_density(ids, i)
    # print sp_den

    p_temp_den = compute_temporal_density(ids)
    # print p_temp_den

    soc_att = compute_soc_att(ids)
    # print soc_att

    w = np.ones(6)

    quality = w[0]*p_quality + w[1]*a_pop + w[2]*a_pop_con + \
              w[3]*sp_den + w[4]*p_temp_den + w[5]*soc_att


    return quality

def _classify_images(X):

    X = scaler_model.transform(X)
    
    predictions = gmm_model.predict(X)

    return predictions

def classify_images(dataset_path, dump_path, gmm_3d):

    n_samples = geo_list.shape[0]

    class_ids = None

    if n_samples > 10 :

        if gmm_3d == True:
            X = gmm_mod.find_points(geo_list, time_list)
        else:
            X = geo_list

        # print geo_list.shape, X.shape
    
        class_ids = _classify_images(X)

    return class_ids

def dump_rec_results(quality_scores, dump_path, ext):

    # sort the ratings
    # np.set_printoptions(suppress=True, precision=3)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    rec_vp_ids = np.argsort(quality_scores)[::-1]
    # print rec_vp_ids
    # print comp_quality

    means = gmm_model.means_

    mean_info = scaler_model.inverse_transform(means)
    # print mean_info 

    # dump the recommendation for later evaluation
    rec_file = dump_path + ext + '_rec.results'
    np.savetxt(rec_file, mean_info , fmt='%f')


def process_dataset(dataset_path, dump_path, gmm_3d=False):
    # set globals first
    if gmm_3d == True:
        set_global_vars(dataset_path, dump_path, 'time')
    else:
        set_global_vars(dataset_path, dump_path, 'basic')
        
    # classify images 
    img_class_ids = classify_images(dataset_path, dump_path, gmm_3d)

    # print img_class_ids

    # iterate for all the gmm components and find quality
    n_components = 10
    comp_quality = np.zeros(n_components)
    for i in range(n_components):
        # find the index of images for ith component
        comp_ids = np.where(img_class_ids == i)[0]
        
        comp_quality[i] = compute_vp_quality(comp_ids, i)

    # find and dump the geolocations and time for recommendation
    if gmm_3d == True:
        dump_rec_results(comp_quality, dump_path, 'time')
    else:
        dump_rec_results(comp_quality, dump_path, 'basic')


    reset_global_vars()

    return

