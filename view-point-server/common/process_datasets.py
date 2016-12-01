import sys
import time
import os
import glob
import numpy as np

import shutil
import main


def generateDB(image_src, dump_path):

    my_object = main.SalientObjectDetection(image_src)

    seg_dumps = dump_path + "segment_dumps/"
    if not os.path.exists(seg_dumps):
        os.makedirs(seg_dumps)
    my_object.process_segments(seg_dumps)

    map_dumps = dump_path + "map_dumps/"
    if not os.path.exists(map_dumps):
        os.makedirs(map_dumps)
    my_object.plot_maps(map_dumps)


def process_dataset(dataset_path, dump_path):
    # browsing the directory

    image_dir = dataset_path + "ImageDB/"
    
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    # Copy required files to dump path
    image_list = dataset_path + "image.list"
    shutil.copy(image_list, dump_path)
    img_details = dataset_path + "images.details"
    shutil.copy(img_details, dump_path)
    img_aesthetic_score = dataset_path + "aesthetic.scores"
    shutil.copy(img_aesthetic_score, dump_path)
    weather_data = dataset_path + "weather.info"
    shutil.copy(weather_data, dump_path)
    geo_info = dataset_path + "geo.info"
    shutil.copy(geo_info, dump_path)
    cam_info = dataset_path + "camera.settings"
    shutil.copy(cam_info, dump_path)

    fp_image_list = open(image_list, 'r')

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        infile = image_dir + image_name

        timer = time.time()

        generateDB(infile, dump_path)
        
        print "Total run time = ", time.time() - timer


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path dump_path"
        sys.exit(0)

    dataset_path = sys.argv[1]
    dump_path = sys.argv[2]

    process_dataset(dataset_path, dump_path)


