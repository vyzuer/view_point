from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import os, sys
import time
from sklearn.externals import joblib
from skimage.feature import local_binary_pattern, multiblock_lbp
from sklearn import preprocessing
import scipy
import cv2
from skimage.feature import hog
from skimage import color
import shutil

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

def xHSVMoments(image):
    img_hsv = color.rgb2hsv(image)

    hsvMom = []
    for i in range(3):
        item = img_hsv[:,:,i]
        mean = np.mean(item)
        variance = np.std(item)
        skewness = scipy.stats.skew(np.reshape(item,-1))

        hsvMom.extend([mean, variance, skewness])

    return hsvMom

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

    nBins = 8

    n_blocks = 4

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5,5), 0)
    
    num_x, num_y = image.shape

    x_step, y_step = 1.0*num_x/n_blocks, 1.0*num_y/n_blocks

    hogHist = []

    for i in range(n_blocks):
        for j in range(n_blocks):
            x_start = int(i*x_step)
            y_start = int(j*y_step)
            x_end = x_start + int(x_step)
            y_end = y_start + int(y_step)

            img = image[x_start:x_end, y_start:y_end]

            fdescriptor = hog(img, orientations=nBins, pixels_per_cell=(8, 8),
                                cells_per_block=(1, 1), visualise=False)

            fd = np.reshape(fdescriptor, (-1, nBins))
            hist = np.sum(fd, axis=0)

            hogHist.extend(hist)


    return hogHist

def xLBP(image):

    # settings for LBP
    radius = 2
    n_points = 8 * radius
    METHOD = 'uniform'
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    n_blocks = 4

    num_x, num_y = image.shape

    x_step, y_step = 1.0*num_x/n_blocks, 1.0*num_y/n_blocks

    lbpHist = []

    for i in range(n_blocks):
        for j in range(n_blocks):
            x_start = int(i*x_step)
            y_start = int(j*y_step)
            x_end = x_start + int(x_step)
            y_end = y_start + int(y_step)

            img = image[x_start:x_end, y_start:y_end]

            lbp = local_binary_pattern(img, n_points, radius, METHOD)
            # n_bins = lbp.max() + 1
            n_bins = 18
            hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))

            lbpHist.extend(hist)

    return lbpHist

def find_features(img):
    fv = []

    hogHist = xHOGHist(img)
    hogHist = np.asarray(hogHist)
    cv2.normalize(hogHist, hogHist, 0, 1, cv2.NORM_MINMAX)
    fv.extend(hogHist)

    hsvMoments = xHSVMoments(img)
    hsvMoments = np.asarray(hsvMoments)
    cv2.normalize(hsvMoments, hsvMoments, 0, 1, cv2.NORM_MINMAX)
    fv.extend(hsvMoments)

    lbp = xLBP(img)
    lbp = np.asarray(lbp)
    cv2.normalize(lbp, lbp, 0, 1, cv2.NORM_MINMAX)
    fv.extend(lbp)

    fv = np.reshape(fv, -1)

    return fv

def process_dataset(db_path, dump_path):

    assert os.path.exists(db_path)

    ascore_list = db_path + '/aesthetic.scores'
    scores = np.loadtxt(ascore_list)

    image_list = db_path + '/image.list'
    image_dir = db_path + "/ImageDB/"
    fp_image_list = open(image_list, 'r')

    a_file = dump_path + '/aesthetic.scores'
    fv_file = dump_path + '/features.list'
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    try:
        os.remove(fv_file)
        os.remove(a_file)
    except OSError:
        pass

    fp = open(fv_file, 'a')
    fp1 = open(a_file, 'a')

    i = 0

    for image_name in fp_image_list:
        image_name = image_name.rstrip("\r\n")
        infile = image_dir + image_name

        print infile

        timer = time.time()

        image = io.imread(infile)

        if image.ndim > 2:

            fv = find_features(image)

            np.savetxt(fp, fv.reshape((1, -1)), fmt='%.8f')
            np.savetxt(fp1, [scores[i]], fmt='%f')
        
        print "Total run time = ", time.time() - timer

        i += 1

    fp_image_list.close()
    fp.close()
    fp1.close()


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print "usage : image_path"
        sys.exit(0)

    img_path = sys.argv[1]

    fv = find_features(img_path)

