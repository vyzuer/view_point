import sys
import time
import os
import glob
import numpy as np
import shutil

from sklearn.externals import joblib


def classify_segments(image_dump_path, cluster_model, lmo_importance, img_id):

    f_feature_list = image_dump_path + "/feature.list"

    feature_list = np.loadtxt(f_feature_list, ndmin=2)
    n_segments = len(feature_list)

    for i in range(n_segments):
        lmo_id = cluster_model.predict(feature_list[i])

        # modify lmo matrix
        lmo_importance[img_id][lmo_id[0]] += 1


def process_dataset(model_path, dump_path):
    
    # load cluster model
    f_model_path = model_path + "cluster_model/cluster.pkl"
    cluster_model = joblib.load(f_model_path)
    seg_dumps = dump_path + "segment_dumps/"

    image_list = dump_path + "image.list"
    fp_image_list = open(image_list, 'r')
    num_images = sum(1 for line in fp_image_list)
    fp_image_list.close()

    fp_image_list = open(image_list, 'r')
    # matrix for lmo importance
    num_lmo = cluster_model.cluster_centers_.shape[0]
    lmo_importance = np.zeros(shape=(num_images, num_lmo), dtype='int')

    i = 0
    # Browse through each image
    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        image_name = os.path.splitext(image_name)[0]
        image_dump_path = seg_dumps + image_name

        # timer = time.time()
        # print image_dump_path

        classify_segments(image_dump_path, cluster_model, lmo_importance, i)
        
        # print "Total run time = ", time.time() - timer

        i += 1


    # find lmo importance
    # dump bag of visual words here for future use
    img_codebook = dump_path + 'img_codebook.list'
    np.savetxt(img_codebook, lmo_importance, fmt='%d')

    fp_image_list.close()
