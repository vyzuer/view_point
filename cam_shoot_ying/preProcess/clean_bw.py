from skimage import io
import glob
import os
import numpy as np
import shutil

def filter_rgb(w_dir, image_db, file_list):
    dst = w_dir + "images.details_0"
    shutil.copy2(file_list, dst)
    f = open(file_list)
    lines = f.readlines()
    f.close()

    n_max = len(lines)

    f_name = w_dir + "image.list"
    fp = open(file_list, 'w')
    fp2 = open(f_name, 'w')
    fp.write("%s" % lines[0])
    for i in xrange(1, n_max):
        file_name = lines[i].split()[0]
        infile = os.path.join(image_db, file_name)
        print infile
        img = io.imread(infile)
        dim = len(img.shape)
        if dim == 3:
            fp.write("%s" % lines[i])
            fp2.write("%s\n" % file_name)
            

    fp.close()
    fp2.close()

if __name__ == '__main__':
    w_dir = '/home/vyzuer/Project/Flickr-YsR/floatMarina/'
    image_db = w_dir + 'ImageDB_640/'
    file_list = w_dir + 'images.details'

    filter_rgb(w_dir, image_db, file_list)

