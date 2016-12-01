import sys
import os
import numpy as np
import pymongo
from pymongo import MongoClient

def process(src_file, dst_file):
    X1 = np.loadtxt(src_file, dtype='string')
    X2 = np.loadtxt(dst_file, dtype='string')

    num1, dim1 = X1.shape
    num2, dim2 = X2.shape

    X = []
    
    X.append(X1[0])
    for i in range(1, num2):
        X.append(X2[i])
        X.append(X1[i])

    for i in range(num2, num1):
        X.append(X1[i])

    np.savetxt(src_file, X, fmt='%s')


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage : dataset_path 1 dataset_path 2"
        sys.exit(0)

    db1 = sys.argv[1]
    db2 = sys.argv[2]

    src1 = db1 + '/photo.url'
    src2 = db2 + '/photo.url'

    process(src1, src2)

