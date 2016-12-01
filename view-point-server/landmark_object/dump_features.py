import sys
import time
import os
import glob
import numpy as np
import shutil

from common import salient_object_detection as obj_detection


def generateDB(image_src, dump_path, seg_dumps, seg_dumps_combined):

    my_object = obj_detection.SalientObjectDetection(image_src, max_iter_slic=1000)

    my_object.process_segments(seg_dumps, seg_dumps_combined, seg_dump=True)

    # map_dumps = dump_path + "map_dumps/"
    # if not os.path.exists(map_dumps):
    #     os.makedirs(map_dumps)
    # my_object.plot_maps(map_dumps)


def process_dataset(dataset_path, dump_path):
    # browsing the directory

    image_dir = dataset_path + "ImageDB/"
    
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    seg_dumps = dump_path + "segment_dumps/"
    seg_dumps_combined = dump_path + "segments/"

    if not os.path.exists(seg_dumps):
        os.makedirs(seg_dumps)
    if not os.path.exists(seg_dumps_combined):
        os.makedirs(seg_dumps_combined)
    else:
        shutil.rmtree(seg_dumps_combined)
        os.makedirs(seg_dumps_combined)

    image_list = dataset_path + "image.list"
    shutil.copy(image_list, dump_path)

    fp_image_list = open(image_list, 'r')

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        infile = image_dir + image_name

        if not os.path.exists(infile):
            continue

        timer = time.time()

        generateDB(infile, dump_path, seg_dumps, seg_dumps_combined)
        
        print "Total run time = ", time.time() - timer

def generateDB_all(image_src, dump_path, seg_dumps, fp_face):

    my_object = obj_detection.SalientObjectDetection(image_src, max_iter_slic=200)

    my_object.process_segments(seg_dumps)

    num_faces = my_object.num_of_faces()
    fp_face.write('%d\n' %(num_faces))



def process_dataset_all(dataset_path, dump_path):
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
    time_info = dataset_path + "time.info"
    shutil.copy(time_info, dump_path)
    cam_info = dataset_path + "camera.settings"
    shutil.copy(cam_info, dump_path)

    seg_dumps = dump_path + "segment_dumps/"

    if not os.path.exists(seg_dumps):
        os.makedirs(seg_dumps)

    image_list = dataset_path + "image.list"
    shutil.copy(image_list, dump_path)

    fp_image_list = open(image_list, 'r')

    face_info = dump_path + "face.list"
    fp_face = open(face_info, "w")

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\n")
        infile = image_dir + image_name

        timer = time.time()

        generateDB_all(infile, dump_path, seg_dumps, fp_face)
        
        print "Total run time = ", time.time() - timer

    fp_image_list.close()
    fp_face.close()

