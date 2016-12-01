from skimage import io
import glob
import os, sys
import numpy as np
import shutil

def filter_rgb(w_dir, image_db, file_list):
    dst = w_dir + "/image.list_1"
    shutil.copy2(file_list, dst)
    f = open(file_list)
    lines = f.readlines()
    f.close()

    n_max = len(lines)

    fp = open(file_list, 'w')
    fp.write("%s" % lines[0])
    for i in xrange(1, n_max):
        file_name = lines[i].split()[0]
        infile = os.path.join(image_db, file_name)
        print infile
        img = io.imread(infile)
        dim = len(img.shape)
        if dim == 3:
            fp.write("%s" % lines[i])
            

    fp.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage : dataset_path"
        sys.exit(0)

    w_dir = sys.argv[1]

    # w_dir = '/home/vyzuer/Project/Flickr-YsR/floatMarina/'
    image_db = w_dir + '/ImageDB/'
    file_list = w_dir + '/image.list'

    # print w_dir
    filter_rgb(w_dir, image_db, file_list)

