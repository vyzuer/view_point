from skimage import io
import glob
import os, sys
import numpy as np
import shutil

def find_mimax(db_dir):
    w_info = db_dir + '/weather.info'
    
    w_details = np.loadtxt(w_info)

    w_min = np.amin(w_details, axis=0)
    w_max = np.amax(w_details, axis=0)

    file_name = db_dir + '/weather.minmax'
    fp = open(file_name, 'a')

    np.savetxt(fp, w_min.reshape((-1,13)), fmt='%.8f')
    np.savetxt(fp, w_max.reshape((-1,13)), fmt='%.8f')

    fp.close()



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage : dataset_path"
        sys.exit(0)

    db_dir = sys.argv[1]

    find_mimax(db_dir)

