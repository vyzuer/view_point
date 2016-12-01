from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from sklearn.externals import joblib
from sklearn import preprocessing
import scipy
import shutil

from common import salient_object_detection as obj_detection

img_features = None
img_codebook = None

def find_images_in_geopixel(src_lat_idx, src_lon_idx, lat_idx, lon_idx, db_path):

    idx_list = []

    num_images = len(lat_idx)

    for i in range(num_images):
        if src_lat_idx == lat_idx[i] and src_lon_idx == lon_idx[i]:
            idx_list.append(i)

    return np.asarray(idx_list)


def images_similar(id_src, id_dst):
    src_fv = img_features[id_src]
    dst_fv = img_features[id_dst]

    src_cb = img_codebook[id_src]
    dst_cb = img_codebook[id_dst]

    fv_dist = np.linalg.norm(src_fv-dst_fv)
    cb_dist = np.linalg.norm(src_cb-dst_cb)

    # print fv_dist, cb_dist

    similar = False
    if fv_dist < 0.70 or cb_dist < 0.70:
        similar = True

    return similar

def find_similar_images(idx, img_list):

    idx_list = []

    for img in img_list:
        if images_similar(idx, img):
            idx_list.append(img)

    return np.asarray(idx_list)


def find_src_image_list(idx, lat_idx, lon_idx, db_path):

    src_lat_idx = lat_idx[idx]
    src_lon_idx = lon_idx[idx]

    src_img_list = find_images_in_geopixel(src_lat_idx, src_lon_idx, lat_idx, lon_idx, db_path)

    # print src_img_list.size

    src_similar_img_list = find_similar_images(idx, src_img_list)

    # print src_similar_img_list.size

    assert src_similar_img_list.size != 0

    return src_similar_img_list

def find_dst_image_list(reco_pos, lat_idx, lon_idx, db_path, g_shape, idx):
    # p_lat, p_lon = np.unravel_index(reco_pos, g_shape)
    p_lat, p_lon = reco_pos[0], reco_pos[1]
    # find images from this geo pixel
    dst_img_list = find_images_in_geopixel(p_lat, p_lon, lat_idx, lon_idx, db_path)

    # print dst_img_list.size

    dst_similar_img_list = find_similar_images(idx, dst_img_list)
    # assert dst_img_list.size != 0

    # print dst_similar_img_list.size

    return dst_similar_img_list

def find_geo_pixel_score(img_idx_list, db_path):
    f_ascore = db_path + '/aesthetic.scores'

    a_score = np.loadtxt(f_ascore)

    score = 0.0
    for img_idx in img_idx_list:
        score += a_score[img_idx]

    return score/a_score.size


def dump_images(img_list_idx, db_path, img_list, dump_dir, dump_name):

    dump_dir = dump_dir + '/' + dump_name
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    for i in img_list_idx:
        img_name = img_list[i]

        src = db_path + '/ImageDB/' + img_name
        dst = dump_dir + '/' + img_name

        os.symlink(src, dst)


def compute_prec(rel_list, p):
    avg_p = 0.0

    for i in range(0, p):
        avg_p += rel_list[i]

    return avg_p/p

def compute_ndcg(rel_list, p):
    ndcg = 0.0

    for i in range(0, p):
        ndcg += (np.power(2, rel_list[i]) -1)/np.log2(i+2)

    idcg = 0.0

    sorted_rel_list = np.sort(rel_list)[::-1]

    for i in range(0, p):
        idcg += (np.power(2, sorted_rel_list[i]) -1)/np.log2(i+2)

    if idcg != 0:
        ndcg /= idcg

    return ndcg

def evaluate_rec(idx, lat_idx, lon_idx, dataset_path, dump_path, features_dump, test_path, rec_type, rec_geo_info, g_shape, top_k=5):

    f_image_list = dataset_path + '/image.list'
    img_list = np.loadtxt(f_image_list, dtype='string')

    # find all the images in the source geo-pixel with similar view
    src_img_idx_list = find_src_image_list(idx, lat_idx, lon_idx, dataset_path)

    # dump the src images
    input_img_name = img_list[idx]

    img_name = os.path.splitext(input_img_name)[0]

    dump_dir = dump_path + '/reco_dump/'

    dump_dir = dump_dir + rec_type + '/' + img_name 

    # if os.path.exists(dump_dir):
    #    shutil.rmtree(dump_dir)

    # os.makedirs(dump_dir)

    src = dataset_path + '/ImageDB/' + input_img_name
    dst = dump_dir + '/' + input_img_name

    # os.symlink(src, dst)

    # dump_images(src_img_idx_list, db_src, img_list, dump_dir, 'src')
    
    src_score = find_geo_pixel_score(src_img_idx_list, dataset_path)

    dst_score =  np.zeros(top_k)
    dst_rel =  np.zeros(top_k)

    dst_img_idx_list = None
    # dump_images(dst_img_idx_list, db_src, img_list, dump_dir, 'dst')
    for i in range(top_k):
        dst_img_idx_list = find_dst_image_list(rec_geo_info[i], lat_idx, lon_idx, dataset_path, g_shape, idx)
        dst_rel[i] = find_geo_pixel_score(dst_img_idx_list, dataset_path)

        if src_score < dst_rel[i]:
            dst_score[i] = 1

    p1 = compute_prec(dst_score, 1)
    p2 = compute_prec(dst_score, 2)
    p5 = compute_prec(dst_score, 5)

    r1 = compute_ndcg(dst_rel, 1)
    r2 = compute_ndcg(dst_rel, 2)
    r5 = compute_ndcg(dst_rel, 5)

    return p1, p2, p5, r1, r2, r5

def reset_global_vars():
    global img_features
    global img_codebook

    img_features = None
    img_codebook = None

def set_global_vars(db_path):
    global img_features
    global img_codebook

    f_img_features = db_path + '/features.list'
    f_img_codebook = db_path + '/img_codebook.list'

    img_features = np.loadtxt(f_img_features)
    img_codebook = np.loadtxt(f_img_codebook)

    # normalize data
    img_features = preprocessing.normalize(img_features, norm='l2')
    img_codebook = preprocessing.normalize(img_codebook, norm='l2')



def find_geo_density(db_path):

    assert os.path.exists(db_path)

    # make a matrix map of this location
    geo_minmax_list = db_path + "geo_minmax.list"
    geo_minmax = np.loadtxt(geo_minmax_list)

    g_xmin, g_xmax = geo_minmax[0], geo_minmax[1]
    g_ymin, g_ymax = geo_minmax[2], geo_minmax[3]
    g_xstep, g_ystep = geo_minmax[4], geo_minmax[5]
    num_xstep, num_ystep = int((g_xmax - g_xmin)/g_xstep), int((g_ymax - g_ymin)/g_ystep)

    f_geo = db_path + "/geo.info"
    geo_list = np.loadtxt(f_geo)

    density = 1.0*geo_list.shape[0]/(num_xstep*num_ystep)

    print density

    density_file = db_path + '/density.info'
    fp = open(density_file, 'w')
    fp.write('%f' %(density))
    fp.close()


def dump_reco_map(rec_geo, dump_path, num_xstep, num_ystep, rec_type):
    plot_dump_path = dump_path + rec_type + '_rec_map.png'

    plot_map = np.zeros(shape=(num_xstep, num_ystep))
    num_components = 10
    for i in range(num_components):
        lat, lon = rec_geo[i,0], rec_geo[i,1]
        plot_map[lat, lon] = np.exp(num_components-i)

    plt.matshow(np.rot90(np.log(plot_map.T+1)/np.log(10)), cmap=plt.cm.gist_earth_r)
    plt.axis('off')
    plt.savefig(plot_dump_path, dpi=200)
    plt.close()


def process_dataset(dataset_path, dump_path, features_dump, test_path, rec_type="basic"):

    assert os.path.exists(dataset_path)
    assert os.path.exists(dump_path)
    assert os.path.exists(features_dump)
    assert os.path.exists(test_path)

    # load global features for future use
    set_global_vars(features_dump)

    # make a matrix map of this location
    geo_minmax_list = features_dump + "geo_minmax.list"
    geo_minmax = np.loadtxt(geo_minmax_list)

    g_xmin, g_xmax = geo_minmax[0], geo_minmax[1]
    g_ymin, g_ymax = geo_minmax[2], geo_minmax[3]
    g_xstep, g_ystep = geo_minmax[4], geo_minmax[5]
    num_xstep, num_ystep = int((g_xmax - g_xmin)/g_xstep), int((g_ymax - g_ymin)/g_ystep)

    f_geo = dataset_path + "/geo.info"
    geo_list = np.loadtxt(f_geo)

    hist, e_lat, e_lon = np.histogram2d(geo_list[:,0], geo_list[:,1], bins=[num_xstep, num_ystep])
    lat_idx = np.digitize(geo_list[:,0], e_lat)-1 # -1 to offset the index for arrays
    lon_idx = np.digitize(geo_list[:,1], e_lon)-1

    res_file = dump_path + '/' + rec_type + '_rec.results'
    rec_geo_info = np.loadtxt(res_file)

    lat_id = np.digitize(rec_geo_info[:,0], e_lat)-1
    lon_id = np.digitize(rec_geo_info[:,1], e_lon)-1
    rec_geo = np.transpose(np.array([lat_id, lon_id]))

    # dump the reco on map
    dump_reco_map(rec_geo, dump_path, num_xstep, num_ystep, rec_type)

    return

    f_id_list = test_path + '/idx.list'
    idx_list = np.loadtxt(f_id_list, dtype='int')

    rec_score = 0.0
    result_file = dump_path + '/' + rec_type + '.ndcg'
    fp = open(result_file, 'w')

    p1, p2, p5 = 0.0, 0.0, 0.0
    r1, r2, r5 = 0.0, 0.0, 0.0

    for i in range(idx_list.size):
        t1, t2, t5, k1, k2, k5 = evaluate_rec(idx_list[i], lat_idx, lon_idx, dataset_path, dump_path, features_dump, test_path, rec_type, rec_geo, hist.shape)
        p1 += t1
        p2 += t2
        p5 += t5
        r1 += k1
        r2 += k2
        r5 += k5

    n_samples = idx_list.size

    p1 /= n_samples
    p2 /= n_samples
    p5 /= n_samples

    r1 /= n_samples
    r2 /= n_samples
    r5 /= n_samples

    print rec_type
    print ("p1: %f\tp2: %f\t p5: %f" % (p1, p2, p5))
    print ("n1: %f\tn2: %f\t n5: %f" % (r1, r2, r5))

    fp.write("p1: %f\tp2: %f\tp5: %f\n" % (p1, p2, p5))
    fp.write("n1: %f\tn2: %f\tn5: %f\n" % (r1, r2, r5))
    fp.close()

    reset_global_vars()


