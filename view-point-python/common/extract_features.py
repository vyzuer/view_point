from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from sklearn.externals import joblib
from sklearn import preprocessing
import scipy
import cv2
from skimage.feature import hog

def xSurfHist(image):
    
    nBins = 64
    hessian_threshold = 500
    nOctaves = 4
    nOctaveLayers = 2

    imageGS = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF(hessian_threshold, nOctaves, nOctaveLayers, False, True)
    keypoints, descriptors = surf.detectAndCompute(imageGS, None) 
    
    surfHist = np.zeros(nBins)

    if len(keypoints) > 0:
        surfHist = np.sum(descriptors, axis=0)

    return surfHist

def xRGBHist(image):
    numBins = 32

    bCh, gCh, rCh = cv2.split(image)

    bins = np.arange(numBins).reshape(numBins,1)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]
    
    rgbHist = []
    for item,col in zip([bCh, gCh, rCh],color):
        hist_item = cv2.calcHist([item],[0],None,[numBins],[0,255])
        rgbHist.extend(hist_item)

    return rgbHist

def xHOGHist(image):

    nBins = 64

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.GaussianBlur(image, (5,5), 0)
    
    fdescriptor = hog(imageBlur, orientations=nBins, pixels_per_cell=(8, 8),
                                cells_per_block=(1, 1), visualise=False)
    

    fd = np.reshape(fdescriptor, (-1, nBins))
    fHist = np.sum(fd, axis=0)

    return fHist


def find_features(image):
    fv = []

    img = io.imread(image)

    surfHist = xSurfHist(img)
    surfHist = np.asarray(surfHist)
    cv2.normalize(surfHist, surfHist, 0, 1, cv2.NORM_MINMAX)
    fv.extend(surfHist)

    hogHist = xHOGHist(img)
    hogHist = np.asarray(hogHist)
    cv2.normalize(hogHist, hogHist, 0, 1, cv2.NORM_MINMAX)
    fv.extend(hogHist)

    rgbHist = xRGBHist(img)
    rgbHist = np.asarray(rgbHist)
    cv2.normalize(rgbHist, rgbHist, 0, 1, cv2.NORM_MINMAX)
    fv.extend(rgbHist)

    fv = np.reshape(fv, -1)

    return fv

def process_dataset(db_path, dump_path):

    assert os.path.exists(dump_path)

    image_list = db_path + '/image.list'
    image_dir = db_path + "/ImageDB/"
    fp_image_list = open(image_list, 'r')

    fv_file = dump_path + '/features.list'
    try:
        os.remove(fv_file)
    except OSError:
        pass

    fp = open(fv_file, 'a')

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\r\n")
        infile = image_dir + image_name

        print infile

        timer = time.time()
        fv = find_features(infile)

        np.savetxt(fp, fv.reshape((1, -1)), fmt='%.8f')
        
        print "Total run time = ", time.time() - timer

    fp_image_list.close()
    fp.close()

