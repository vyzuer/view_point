#!/bin/csh

#set dump_path = "/home/yogesh/Project/Flickr-YsR/small_merlion/ImageDB/"
#set dataset_path = "/home/yogesh/Project/Flickr-YsR/merlionDB_1/DB/0/ImageDB/"

# set dump_path = "/home/yogesh/Project/Flickr-YsR/merlionImages/top_N_images/ImageDB_640/"
# set dataset_path = "/home/yogesh/Project/Flickr-YsR/merlionImages/top_N_images/ImageDB/"

# set dataset_path = "/home/yogesh/Copy/Flickr-code/PhotographyAssistance/testing/images/"
# set dump_path = "/home/yogesh/Copy/Flickr-code/PhotographyAssistance/testing/images_0/"
# set dataset_path = "/home/yogesh/Project/Flickr-YsR/floatMarina/ImageDB/"
# set dump_path = "/home/yogesh/Project/Flickr-YsR/floatMarina/ImageDB_640/"

# set dataset_path = '/home/vyzuer/Project/Flickr-code/pos_results/ImageDB_1/'
# set dump_path = '/home/vyzuer/Project/Flickr-code/pos_results/ImageDB/'

set dump_path = "/mnt/project/VP/ODB/Holidays/ImageDB/"
set dataset_path = "/mnt/project/VP/ODB/Holidays/ImageDB_Orig/"

mkdir -p $dump_path

# resizing images in dataset
python resize_image.py $dataset_path $dump_path

