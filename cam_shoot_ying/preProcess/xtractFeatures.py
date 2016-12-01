import Image
import ImageOps
import numpy as np
from numpy import linalg
import matplotlib
from matplotlib import pyplot as plt
import os
import glob
import sys
import cv
import cv2
import math
from skimage.feature import hog, ORB
from skimage import data, color, exposure

class xtractFeatures:
    def __init__(self, sPath, surf=False, hog=False, orb=False, rgb=False):
        self.sPath = sPath
        self.dbPath = sPath + 'ImageDB/'
        self.imageList = sPath + 'image.list'
        self.surf = surf
        self.hog = hog
        self.orb = orb
        self.rgb = rgb

    def __cleanImageList(self):
        fpImageList = open(self.imageList, 'r')
        fName = self.sPath + 'image_clean.list'

        fp = open(fName, 'w')

        for image_src in fpImageList:
            image = self.dbPath + image_src
            if cv2.imread(image):
                fp.write("%s\n" % image_src)

        fp.close()
    
    def xtract(self):
        fpImageList = open(self.imageList, 'r')
    
        fName = self.sPath + 'feature.list'
        fp = open(fName, 'w')

        for image_src in fpImageList:
            image_src = image_src.rstrip("\n")
            image_src = self.dbPath + image_src
            print image_src
            fv = []
            if self.surf == True:
                surfHist = self.__xSurfHist(image_src)
                surfHist = np.asarray(surfHist)
                cv2.normalize(surfHist, surfHist, 0, 1, cv2.NORM_MINMAX)
                fv.extend(surfHist)
                
            if self.hog == True:
                hogHist = self.__xHOGHist(image_src)
                hogHist = np.asarray(hogHist)
                cv2.normalize(hogHist, hogHist, 0, 1, cv2.NORM_MINMAX)
                fv.extend(hogHist)

            if self.orb == True:
                orbDes = self.__xORB(image_src)
                orbDes = np.asarray(orbDes)
                cv2.normalize(orbDes, orbDes, 0, 1, cv2.NORM_MINMAX)
                fv.extend(orbDes)

            if self.rgb == True:
                rgbHist = self.__xRGBHist(image_src)
                rgbHist = np.asarray(rgbHist)
                cv2.normalize(rgbHist, rgbHist, 0, 1, cv2.NORM_MINMAX)
                fv.extend(rgbHist)

            fv = np.reshape(fv, -1)
            for i in fv:
                fp.write("%s " % i)
            fp.write("\n")

        fp.close()
        fpImageList.close()

    def __xRGBHist(self, image_src):
        numBins = 32
    
        image = cv2.imread(image_src)
        bCh, gCh, rCh = cv2.split(image)

        bins = np.arange(numBins).reshape(numBins,1)
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
        
        rgbHist = []
        for item,col in zip([bCh, gCh, rCh],color):
            hist_item = cv2.calcHist([item],[0],None,[numBins],[0,255])
            rgbHist.extend(hist_item)

        return rgbHist
    
    def __xHOGHist(self, image_src):
    
        nBins = 72

        image = cv2.imread(image_src, 0)
        imageBlur = cv2.GaussianBlur(image, (5,5), 0)
        
        fdescriptor = hog(imageBlur, orientations=nBins, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), visualise=False)
        
        idx = 0
        count = 1
        fHist = np.zeros(nBins)
        for val in fdescriptor:
            fHist[idx] += val
            count += 1
            idx += 1
            if count%nBins == 1:
                idx = 0

        return fHist
    
    def __xSurfHist(self, image_src):
    
        nBins = 64
        hessian_threshold = 500
        nOctaves = 4
        nOctaveLayers = 2
    
        image = cv2.imread(image_src)
        imageGS = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        surf = cv2.SURF(hessian_threshold, nOctaves, nOctaveLayers, False, True)
        keypoints, descriptors = surf.detectAndCompute(imageGS, None) 
        
        surfHist = np.zeros(nBins)

        if len(keypoints) > 0:
    
            surfHist = np.zeros(nBins)
            for val in descriptors:
                idx = 0
                rowFeatures = np.array(val, dtype = np.float32)
                for val2 in rowFeatures:
                    surfHist[idx] += val2
                    idx += 1
    
        return surfHist

    def __xORB(self, image_src):
        nBins = 100
        image = cv2.imread(image_src, 0)
        detector_extractor = ORB(n_keypoints=nBins)
        detector_extractor.detect_and_extract(image)

        print detector_extractor.descriptors.shape

        keypoints = detector_extractor.keypoints
        descriptors = detector_extractor.descriptors
        orbHist = np.zeros(nBins)

        if len(keypoints) > 0:
    
            orbHist = np.zeros(nBins)
            for val in descriptors:
                idx = 0
                rowFeatures = np.array(val, dtype = np.float32)
                for val2 in rowFeatures:
                    orbHist[idx] += val2
                    idx += 1

        return orbHist

