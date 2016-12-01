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
qimg_features = None
qimg_codebook = None
num_query_samples = None
num_db_images = None

dataset = None

def euclidean_distance(id_src, id_dst):
    src_fv = qimg_features[id_src]
    dst_fv = img_features[id_dst]

    src_cb = qimg_codebook[id_src]
    dst_cb = img_codebook[id_dst]

    fv_dist = np.linalg.norm(src_fv-dst_fv)
    cb_dist = np.linalg.norm(src_cb-dst_cb)

    # print fv_dist, cb_dist

    dist = fv_dist * cb_dist
    # dist = cb_dist
    # dist = fv_dist

    return dist


def retrieve_images(qid, top_k):
    # iterate over all the images in the dataset
    # to find similar images

    similarity_score = np.zeros(num_db_images)

    for i in range(num_db_images):
        similarity_score[i] = euclidean_distance(qid, i)

    similar_images = np.argsort(similarity_score)

    return similar_images


def set_global_vars(dump_path, qdump_path, db_name):
    global img_features
    global img_codebook
    global qimg_features
    global qimg_codebook
    global num_query_samples
    global num_db_images

    global dataset

    f_img_features = dump_path + '/features.list'
    f_img_codebook = dump_path + '/img_codebook.list'
    f_qimg_features = qdump_path + '/features.list'
    f_qimg_codebook = qdump_path + '/img_codebook.list'

    img_features = np.loadtxt(f_img_features)
    img_codebook = np.loadtxt(f_img_codebook)
    qimg_features = np.loadtxt(f_qimg_features)
    qimg_codebook = np.loadtxt(f_qimg_codebook)

    # normalize data
    img_features = preprocessing.normalize(img_features, norm='l2')
    img_codebook = preprocessing.normalize(img_codebook, norm='l2')
    qimg_features = preprocessing.normalize(qimg_features, norm='l2')
    qimg_codebook = preprocessing.normalize(qimg_codebook, norm='l2')

    num_query_samples = qimg_codebook.shape[0]
    num_db_images = img_codebook.shape[0]

    dataset = db_name

def find_class_id(dump_path):
    img_list = dump_path + 'image.list'
    img_names = np.loadtxt(img_list, dtype='string')
    
    ids = np.zeros(num_db_images, dtype='int')

    for i in range(num_db_images):
        if dataset == 'zubud':
            ids[i] = int(img_names[i][6:10])  # for ZuBuD dataset
        else:
            ids[i] = int(img_names[i][0:4])  # for Holidays dataset

    return ids

def process_dataset(db_path, dump_path, qdump_path, top_k=3):

    assert os.path.exists(db_path)
    assert os.path.exists(dump_path)
    assert os.path.exists(qdump_path)

    db_name = 'zubud'
    db_name = 'holidays'

    # load global features for future use
    set_global_vars(dump_path, qdump_path, db_name)

    # load ground truth
    gt_file = db_path + 'groundtruth.txt'
    gt = np.loadtxt(gt_file, dtype='int', skiprows=1)

    result_file = qdump_path + '/' + 'result.map'
    fp = open(result_file, 'w')

    # first retrieve the class id of db images
    dst_img_classid = find_class_id(dump_path)
    freq_list = np.bincount(dst_img_classid)
    print freq_list

    prec = 0.0
    base = 1000
    if dataset == 'zubud':
        base = 1

    for i in range(num_query_samples):
        img_list = retrieve_images(i, top_k)
        print i, gt[i][1]
        p = 0.0
        for j in range(freq_list[base + i]):   
        # for j in range(top_k):   
            if gt[i][1] == dst_img_classid[img_list[j]]:
                p += 1.0
            print dst_img_classid[img_list[j]]

        prec += p/freq_list[base+i] 
        # prec += p/top_k 


    print 'MAP = ', prec/num_query_samples
    fp.write("MAP : %f" % (prec/num_query_samples))

    fp.close()


