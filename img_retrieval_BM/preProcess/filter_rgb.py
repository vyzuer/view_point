from skimage import io
import glob
import os
import numpy as np

def filter_rgb(w_dir, image_db, file_list):
    f_name = file_list
    fp = open(f_name, 'w')
    for infile in glob.glob(os.path.join(image_db, '*.jpg')):
        (filepath, filename) = os.path.split(infile)
        print filename
        img = io.imread(infile)
        dim = len(img.shape)
        if dim == 3:
            fp.write("%s\n" % filename)
            

    fp.close()

if __name__ == '__main__':
    w_dir = '/home/vyzuer/Project/Flickr-YsR/esplanade/'
    image_db = w_dir + 'ImageDB_640/'
    file_list = w_dir + 'image.list'

    filter_rgb(w_dir, image_db, file_list)

