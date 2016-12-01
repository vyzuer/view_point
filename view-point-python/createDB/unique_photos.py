import sys
import os
import numpy as np
import pymongo
from pymongo import MongoClient

def process(src_file, dst_file, location):
    X = np.loadtxt(src_file, dtype='string')

    num, dim = X.shape

    client = MongoClient()
    db = client.flickr_vp
    col = db[location]

    try:
        os.remove(dst_file)
    except OSError:
        pass

    fp = open(dst_file, 'a')

    for i in range(num):
        photo_id = X[i][1]
        print photo_id
        photo = col.find_one({'photo_id': photo_id})
        if not photo:
            col.insert({'photo_id': photo_id})
            np.savetxt(fp, np.atleast_1d(X[i][None]), fmt='%s', delimiter=" ")
            # fp.write('%s\n' % (X[i]))

    fp.close()

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path location"
        sys.exit(0)

    db = sys.argv[1]
    location = sys.argv[2]

    db_path = db + location

    src_file = db_path + '/photo.url'
    dst_file = db_path + '/photo.url'

    process(src_file, dst_file, location)

