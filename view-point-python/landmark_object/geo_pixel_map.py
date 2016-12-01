import itertools
import sys, os
import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from sklearn.externals import joblib
from sklearn import preprocessing


def dump_geo_pixel_map(geo_list, a_score, e_lat, e_lon, num_xstep, num_ystep, dump_dir, dump_map, s_gmm_model):

    n_samples = a_score.size

    if n_samples < 2:
        print s_gmm_model

    assert n_samples > 1

    lat_idx = np.digitize(geo_list[:,0], e_lat)-1 # -1 to offset the index for arrays
    lon_idx = np.digitize(geo_list[:,1], e_lon)-1

    a_score_map = np.zeros(shape=(num_xstep+1, num_ystep+1))
    i_score_map = np.zeros(shape=(num_xstep+1, num_ystep+1))
    hist = np.zeros(shape=(num_xstep+1, num_ystep+1))

    for i in range(n_samples):
        a_score_map[lat_idx[i]][lon_idx[i]] += a_score[i]
        hist[lat_idx[i]][lon_idx[i]] += 1

    alpha = 1.0/10
    beta = 6.0
    for i in range(num_xstep):
        for j in range(num_ystep):
            if hist[i][j] > 1:
                num_imgs = hist[i][j]
                a_score_map[i][j] /= num_imgs
                # idf = np.exp(beta-alpha*num_imgs)/(1+np.exp(beta-alpha*num_imgs))
                idf = 1+np.exp(alpha*num_imgs-beta)
                i_score_map[i][j] = a_score_map[i][j]/idf

    a_score_map = a_score_map[:num_xstep, :num_ystep]
    i_score_map = i_score_map[:num_xstep, :num_ystep]
    hist = hist[:num_xstep, :num_ystep]

    # a_score_map /= np.max(a_score_map)

    # max_i = np.max(i_score_map)
    # if max_i != 0:
    #     i_score_map /= max_i

    map_dump_path = dump_dir + '/geo_pixel.map'
    np.savetxt(map_dump_path, a_score_map, fmt='%0.8f')

    map_dump_path_i = dump_dir + '/geo_pixel_i.map'
    np.savetxt(map_dump_path_i, i_score_map, fmt='%0.8f')

    if dump_map == True:
        plot_dump_path = dump_dir + '/geo_pixel_map.png'
        plt.matshow(np.rot90(a_score_map.T))
        # plt.matshow(np.rot90(a_score_map.T), cmap=plt.cm.gist_earth_r)
        # plt.matshow(np.rot90(np.log(a_score_map.T+1)/np.log(10)), cmap=plt.cm.gist_earth_r)
        plt.axis('off')
        plt.savefig(plot_dump_path, dpi=100)
        plt.close()

    if dump_map == True:
        plot_dump_path = dump_dir + '/geo_pixel_map_i.png'
        plt.matshow(np.rot90(i_score_map.T))
        # plt.matshow(np.rot90(a_score_map.T), cmap=plt.cm.gist_earth_r)
        # plt.matshow(np.rot90(np.log(a_score_map.T+1)/np.log(10)), cmap=plt.cm.gist_earth_r)
        plt.axis('off')
        plt.savefig(plot_dump_path, dpi=100)
        plt.close()

    if dump_map == True:
        plot_dump_path = dump_dir + '/geo_pixel_hist.png'
        plt.matshow(np.rot90(np.log(hist.T+1)/np.log(10)))
        # plt.matshow(np.rot90(np.log(hist.T+1)/np.log(10)), cmap=plt.cm.gist_earth_r)
        plt.axis('off')
        plt.savefig(plot_dump_path, dpi=100)
        plt.close()

def find_geo_pixel_map_human(dump_path, model_path, e_lat, e_lon, num_xstep, num_ystep, dump_map, ext='geo_pixel_map'):

    s_gmm_model = model_path + "/gmm_models/human_obj/"

    f_face_list = dump_path + "/face.list"
    f_pos_list = dump_path + "/geo.info"
    f_a_score = dump_path + "/aesthetic.scores"

    assert os.path.isfile(f_pos_list)
    assert os.path.isfile(f_a_score)
    assert os.path.isfile(f_face_list)

    a_score = np.loadtxt(f_a_score)
    data = np.loadtxt(f_pos_list)
    face_info = np.loadtxt(f_face_list)
    
    geo_list = data[face_info!=0] 
    a_score = a_score[face_info!=0]

    dump_dir = s_gmm_model + '/' + ext 
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    dump_geo_pixel_map(geo_list, a_score, e_lat, e_lon, num_xstep, num_ystep, dump_dir, dump_map, s_gmm_model)

def find_geo_pixel_map(s_path, s_gmm_model, e_lat, e_lon, num_xstep, num_ystep, dump_map=False, ext='geo_pixel_map'):
    
    f_pos_list = s_path + "/geo.info"
    f_a_score = s_path + "/aesthetic.scores"

    assert os.path.isfile(f_pos_list)
    assert os.path.isfile(f_a_score)

    a_score = np.loadtxt(f_a_score)
    geo_list = np.loadtxt(f_pos_list)

    dump_dir = s_gmm_model + '/' + ext 
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    dump_geo_pixel_map(geo_list, a_score, e_lat, e_lon, num_xstep, num_ystep, dump_dir, dump_map, s_gmm_model)


def process_lmo(model_path, dump_path, dump_map=False):
    assert os.path.exists(model_path)
    assert os.path.exists(dump_path)

    # make a matrix map of this location
    geo_minmax_list = dump_path + "/geo_minmax.list"
    geo_minmax = np.loadtxt(geo_minmax_list)

    g_xmin, g_xmax = geo_minmax[0], geo_minmax[1]
    g_ymin, g_ymax = geo_minmax[2], geo_minmax[3]
    g_xstep, g_ystep = geo_minmax[4], geo_minmax[5]
    num_xstep, num_ystep = int((g_xmax - g_xmin)/g_xstep), int((g_ymax - g_ymin)/g_ystep)
    
    f_geo_list = dump_path + "/geo.info"
    assert os.path.isfile(f_geo_list)
    geo_list = np.loadtxt(f_geo_list)

    hist, e_lat, e_lon = np.histogram2d(geo_list[:,0], geo_list[:,1], bins=[num_xstep, num_ystep])

    f_num_clusters = model_path + "/segments/_num_clusters.info"
    gmm_model_path = model_path + "/gmm_models/"
    clusters_info = dump_path + "/lm_objects/"

    num_clusters = np.loadtxt(f_num_clusters, dtype=np.int)

    for i in range(num_clusters):
    # for i in range(1):
        s_path = clusters_info + str(i)
        s_gmm_model = gmm_model_path + str(i)

        find_geo_pixel_map(s_path, s_gmm_model, e_lat, e_lon, num_xstep, num_ystep, dump_map)

    # dump geopixel map for humans
    find_geo_pixel_map_human(dump_path, model_path, e_lat, e_lon, num_xstep, num_ystep, dump_map)


