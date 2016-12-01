from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import os, sys
import time
from sklearn.externals import joblib
from skimage.feature import local_binary_pattern, multiblock_lbp
from sklearn import preprocessing
import scipy
import cv2
from skimage.feature import hog
from skimage import color
import shutil
import math

def find_direction(geo_info, geo_origin):

    lat , lon = geo_info[0], geo_info[1]
    lat0, lon0 = geo_origin[0], geo_origin[1]

    angle = 0.0

    # first find quadrant
    if lat > lat0:
        if lon > lon0:
            r = np.abs(lat - lat0)/np.abs(lon - lon0)
            angle = math.degrees(math.atan(r))
        else:
            r = np.abs(lat - lat0)/np.abs(lon - lon0)
            angle = 180.0 - math.degrees(math.atan(r))
    else:
        if lon > lon0:
            r = np.abs(lat - lat0)/np.abs(lon - lon0)
            angle = 360.0 - math.degrees(math.atan(r))
        else:
            r = np.abs(lat - lat0)/np.abs(lon - lon0)
            angle = 180.0 + math.degrees(math.atan(r))

    return angle

def process_dataset(db_path, dump_path, a_dump):

    assert os.path.exists(db_path)

    geo_list = db_path + '/geo.info'
    geo = np.loadtxt(geo_list)

    geo_origin = db_path + '/geo.origin'
    geo_o = np.loadtxt(geo_origin)

    a_scores = a_dump + '/aesthetic.scores'
    scores = np.loadtxt(a_scores)

    num_images = geo.shape[0]

    fv_file = dump_path + '/direction.list'
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    try:
        os.remove(fv_file)
    except OSError:
        pass

    fp = open(fv_file, 'a')

    for i in range(num_images):

        direction = find_direction(geo[i], geo_o)

        np.savetxt(fp, [direction], fmt='%f')

    fp.close()

    directions = np.loadtxt(fv_file)

    n_bins = 24
    hist, edges = np.histogram(directions, bins=n_bins, range=(0,360))

    print hist, edges

    img_ids = np.digitize(directions, edges)-1

    top_imgs = np.zeros(n_bins, dtype='int')
    top_scores = np.zeros(n_bins)

    for i in range(num_images):
        img_id = img_ids[i]
        if scores[i] > top_scores[img_id]:
            top_imgs[img_id] = i
            top_scores[img_id] = scores[i]

    print top_scores
    print top_imgs
    print directions[top_imgs]

if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print "usage : dataset_path dump_path a_dump"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]
    a_dump = sys.argv[3]

    fv = process_dataset(dataset_path, dump_path, a_dump)

