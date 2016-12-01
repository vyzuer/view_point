from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.segmentation import relabel_sequential
from skimage import img_as_float
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from matplotlib import pyplot as plt
import numpy as np
import string
from scipy import ndimage
import os
import time
import cv2
from skimage.feature import hog
from skimage import morphology
from sklearn.externals import joblib
from skimage import transform as tf

from multiprocessing import Process, Array
from numpy.core.umath_tests import inner1d

from collections import Counter

import segmentation as seg
import saliency as saliency
import face_detection as face_detection
import cython_utils as cutils

NUM_OF_SALIENT_OBJECTS = 30
FV_LENGTH = 200

x_scale = 4
y_scale = 3

dump_segs = True

class SalientObjectDetection:
    def __init__(self, img, image_src='/tmp/temp.jpg', segment_dump="/tmp/", a_score=1, prediction_stage=False, min_saliency=0.02, max_iter_slic=100):
        self.timer = time.time()
        self.image_src = image_src
        self.a_score = a_score
        self.image = img
        # self.image = io.imread(image_src)
        print "Image source : ", image_src

        self.__set_timer("segmentation...")
        segment_object = seg.SuperPixelSegmentation(self.image, max_iter=max_iter_slic)
        self.segment_map = segment_object.getSegmentationMap()
        self.slic_map = segment_object.getSlicSegmentationMap()
        self.__print_timer("segmentation")

        self.__set_timer("saliency...")
        saliency_object = saliency.Saliency(self.image, 3)
        self.saliency_map = saliency_object.getSaliencyMap()
        self.__print_timer("saliency")

        # perform face detection
        self.__set_timer("face detection...")
        self.faces = face_detection.detect(np.array(self.image))
        self.__print_timer("face detection")

        self.__set_timer("saliency detection of objects...")
        self.saliency_list, self.salient_objects, self.pixel_count, self.segment_map2 = cutils.detect_saliency_of_segments(self.segment_map.astype(np.intp), self.saliency_map, min_saliency)
        self.__print_timer("saliency detection of objects")


    def num_of_faces(self):
        return len(self.faces)

    def classify_objects(self, dump_path, vp_model_path, seg_dump=False):

        img_x, img_y = self.segment_map2.shape
        segs = ndimage.find_objects(self.segment_map2)
        # ignore the last segment as it is for faces
        num_segments = len(segs) - 1

        # 0 is for backgroud
        # last segment is for faces
        # num_segments = len(self.saliency_list) - 1

        seg_path = None
        if seg_dump:
            dir_name = os.path.split(self.image_src)[1]
            dir_name = os.path.splitext(dir_name)[0]

            dir_path = dump_path + "/segments/"

            seg_path = dir_path + dir_name
            if not os.path.exists(seg_path):
                os.makedirs(seg_path)

        model_dump = vp_model_path + "/cluster_model/cluster.pkl"
        cluster_model = joblib.load(model_dump)

        lm_objects_list = []
        saliency_list = []

        j = 1
        for i in xrange(num_segments):

            # for deleted segments
            if segs[i] == None:
                continue

            segment_img = self.image[segs[i]]

            segment_copy = np.copy(segment_img)

            mask = self.segment_map2[segs[i]]
            idx=(mask!=i+1)
            segment_copy[idx] = 255, 255, 255
            
            if (seg_dump == True):
                fig, ax = plt.subplots(1, 2)

                ax[0].axis('off')
                ax[1].axis('off')
                fig.patch.set_visible(False)

                ax[0].imshow(segment_img)
                ax[1].imshow(segment_copy)

                file_name = seg_path + '/' + str(j) + ".png"
                plt.savefig(file_name, dpi=60)
                plt.close()
            
            obj_id = self.__predict_lm_object(segment_copy, mask, i+1, cluster_model)

            lm_objects_list.append(obj_id)
            saliency_list.append(self.saliency_list[i+1])

            j += 1

        return np.asarray(lm_objects_list), np.asarray(saliency_list)

    def __predict_lm_object(self, img_segment, seg_block, idx, cluster_model):
        fv = self.__find_segment_features(img_segment, seg_block, idx)

        obj_id = cluster_model.predict(fv.reshape(1, -1))

        return obj_id


    def process_segments(self, dump_path, master_dump=None, seg_dump=False):
        global dump_segs
        if master_dump == None:
            dump_segs = False

        img_x, img_y = self.segment_map2.shape
        segs = ndimage.find_objects(self.segment_map2)
        # ignore the last segment as it is for faces
        num_segments = len(segs) - 1

        # 0 is for backgroud
        # last segment is for faces
        # num_segments = len(self.saliency_list) - 1

        dir_name = os.path.split(self.image_src)[1]
        dir_name = os.path.splitext(dir_name)[0]

        dir_path = dump_path + dir_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        seg_path = dir_path + "/segments/"   
        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        feature_file = dir_path + "/feature.list"
        saliency_file = dir_path + "/saliency.list"
        pos_file = dir_path + "/pos.list"

        if os.path.isfile(feature_file):
            os.unlink(feature_file)
        if os.path.isfile(saliency_file):
            os.unlink(saliency_file)
        if os.path.isfile(pos_file):
            os.unlink(pos_file)

        fp = open(saliency_file, 'w')
        fp1 = open(feature_file, 'w')
        fp2 = open(pos_file, 'w')

        # master dump
        f_segment_features = None
        f_segment_images = None

        fp3 = None
        fp4 = None

        if dump_segs == True:
            f_segment_features = master_dump + '/segments.list'
            f_segment_images = master_dump + '/png.list'

            fp3 = open(f_segment_features, 'a')
            fp4 = open(f_segment_images, 'a')

        j = 1
        for i in xrange(num_segments):

            # for deleted segments
            if segs[i] == None:
                continue

            fp.write("%0.8f\n" % self.saliency_list[i+1])

            segment_img = self.image[segs[i]]

            # find the position
            x_0 =  segs[i][0].start
            x_1 =  segs[i][0].stop
            y_0 = segs[i][1].start
            y_1 = segs[i][1].stop
            x_pos = (x_1 + x_0)/2.0
            y_pos = (y_1 + y_0)/2.0
            # print segs[i]
            # print '{0} {1}'.format(x_pos, y_pos)

            fp2.write("{0:0.8f} {1:0.8f}\n".format(x_pos/img_x, y_pos/img_y))

            segment_copy = np.copy(segment_img)

            mask = self.segment_map2[segs[i]]
            idx=(mask!=i+1)
            segment_copy[idx] = 255, 255, 255
            
            file_name = seg_path + str(j) + ".png"
            if (seg_dump == True):
                fig, ax = plt.subplots(1, 2)

                ax[0].axis('off')
                ax[1].axis('off')
                fig.patch.set_visible(False)

                ax[0].imshow(segment_img)
                ax[1].imshow(segment_copy)

                plt.savefig(file_name, dpi=60)
                plt.close()
            
            self.__dump_segment_features(segment_copy, mask, fp1, fp3, i+1)

            if dump_segs == True:
                fp4.write("%s\n" % file_name)

            j += 1

        fp.close()
        fp1.close()
        fp2.close()
        if dump_segs == True:
            fp3.close()
            fp4.close()

    def __find_segment_features(self, segment_copy, seg_block, idx):
        fv = []

        # shape of the segment
        scale_factor = 1.0
        img_height, img_width, n_dim = segment_copy.shape
        max_size = np.max([img_height, img_width])
        fv.extend([scale_factor*img_height/max_size, scale_factor*img_width/max_size])
    
        # block wise shape features
        shapeFeature = self.__xShapeFeatures(seg_block, idx)
        shapeFeature = np.asarray(shapeFeature)
        fv.extend(shapeFeature)

        # self.__set_timer("surf")
        surfHist = self.__xSurfHist(segment_copy)
        surfHist = np.asarray(surfHist)
        cv2.normalize(surfHist, surfHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(surfHist)
        # self.__print_timer("surf")

        # self.__set_timer("hog")
        hogHist = self.__xHOGHist(segment_copy)
        hogHist = np.asarray(hogHist)
        cv2.normalize(hogHist, hogHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(hogHist)
        # self.__print_timer("hog")

        # self.__set_timer("rgb")
        rgbHist = self.__xRGBHist(segment_copy, seg_block, idx)
        rgbHist = np.asarray(rgbHist)
        cv2.normalize(rgbHist, rgbHist, 0, 1, cv2.NORM_MINMAX)
        fv.extend(rgbHist)
        # self.__print_timer("rgb")

        fv = np.reshape(fv, -1)

        return fv


    def __dump_segment_features(self, segment_copy, seg_block, fp1, fp2, idx):

        fv = self.__find_segment_features(segment_copy, seg_block, idx)
        np.savetxt(fp1, np.atleast_2d(fv), fmt='%.8f')

        if dump_segs == True:
            np.savetxt(fp2, np.atleast_2d(fv), fmt='%.8f')
        

    def __xShapeFeatures(self, seg_block, idx, num_xblock=12, num_yblock=12):
        shapeFeature = []
        scale = 1.0

        img_height, img_width = seg_block.shape

        x_step = 1.0*img_height/num_xblock
        y_step = 1.0*img_width/num_yblock
        block_size = x_step*y_step

        x, y = 0.0, 0.0

        for i in range(num_xblock):
            y = 0.0
            for j in range(num_yblock):
                x_start = int(x)
                y_start = int(y)
                x_end = int(x+x_step)
                y_end = int(y+y_step)
                img_block = seg_block[x_start:x_end, y_start:y_end]
                pixel_count = img_block.ravel().tolist().count(idx)
                ratio = scale*pixel_count/block_size
                shapeFeature.extend([ratio])
                y += y_step

            x += x_step

        return shapeFeature

    
    def __find_mean_color(self, image_block):
        mean_r = np.mean(image_block[:, :, 0])
        mean_g = np.mean(image_block[:, :, 1])
        mean_b = np.mean(image_block[:, :, 2])

        return [mean_r, mean_g, mean_b]

    def __xRGBHistWrap(self, image, num_xblock=12, num_yblock=12):
        rgbHist = []
        scale = 2

        img_height, img_width, n_dim = image.shape
        image = rgb2lab(image)

        x_step = img_height/num_xblock
        y_step = img_width/num_yblock

        x, y = 0, 0

        for i in range(num_xblock):
            y = 0
            for j in range(num_yblock):
                img_block = image[x:x+x_step, y:y+y_step,:]
                # hist_item = self.__xRGBHist(img_block)
                # hist_item = np.asarray(hist_item)
                # cv2.normalize(hist_item, hist_item, 0, scale, cv2.NORM_MINMAX)
                # rgbHist.extend(hist_item)
                mean_color = self.__find_mean_color(img_block)
                rgbHist.extend(mean_color)
                y += y_step

            x += x_step

        return rgbHist

    
    def __xRGBHist(self, image, seg_block, idx):
        numBins = 256
    
        # seg = np.copy(seg_block).astype(dtype=np.uint8)
        # mask=(seg!=idx)
        # seg[mask] = 0

        bCh, gCh, rCh = cv2.split(image)

        bins = np.arange(numBins).reshape(numBins,1)
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
        
        rgbHist = []
        for item,col in zip([bCh, gCh, rCh],color):
            hist_item = cv2.calcHist([item],[0],None,[numBins],[0,255])
            rgbHist.extend(hist_item)

        return rgbHist

    
    def __xHOGHist(self, image):
    
        nBins = 64

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageBlur = cv2.GaussianBlur(image, (5,5), 0)
        
        fdescriptor = hog(imageBlur, orientations=nBins, pixels_per_cell=(8, 8),
                                    cells_per_block=(1, 1), visualise=False)
        

        fd = np.reshape(fdescriptor, (-1, nBins))
        fHist = np.sum(fd, axis=0)

        return fHist

    
    def __xSurfHist(self, image):
    
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


    def __set_timer(self, mesg=""):
        self.timer = time.time()
        if len(mesg) > 0:
            print "Starting ", mesg, "..."

    def __print_timer(self, mesg=""):
        print mesg, "done. run time = ", time.time() - self.timer


    def __find_saliency_of_segments(self, seg_map, sal_map):
        height, width = seg_map.shape
        num_segments = np.amax(seg_map)+1
        #print num_segments
        saliency_list = np.zeros(shape=(num_segments,2), dtype=(float, int))

        for i in xrange(num_segments):
            saliency_list[i][1] = i

        pixel_count = np.zeros(num_segments)
        for i in xrange(height):
            for j in xrange(width):
                if seg_map[i][j] != 0:
                    seg_id = int(seg_map[i][j])
                    saliency_list[seg_id][0] += sal_map[i][j]
                    pixel_count[seg_id] += 1

        #print sorted(pixel_count)
        #print pixel_count
        
        # for i in xrange(num_segments):
        #     #print pixel_count[i]
        #     #print saliency_list[i][0]
        #     #print saliency_list[i][1]

        #     saliency_list[i][0] = saliency_list[i][0]/(pixel_count[i]+1)

        #print saliency_list
        saliency_list = sorted(saliency_list, key=lambda x: x[0], reverse=True)
        #saliency_list.sort()
        #print saliency_list

        salient_objects = np.zeros(num_segments)
        for i in xrange(num_segments):
            #print saliency_list[i][0]
            #print saliency_list[i][1]
            salient_objects[int(saliency_list[i][1])] = 1
        
        return saliency_list, salient_objects, pixel_count


    def plot_maps(self, db_path):

        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(8, 3, forward=True)
        plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
        
        # fig.patch.set_visible(False)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')

        ax[0].imshow(mark_boundaries(self.image, self.slic_map))
        ax[0].set_title("Image")

        ax[1].imshow(self.saliency_map, interpolation='nearest')
        ax[1].set_title("Saliency")

        ax[2].imshow(mark_boundaries(self.image, self.segment_map))
        ax[2].set_title("Segmented Image")

        ax[3].imshow(self.segment_map2)
        ax[3].set_title("Composition")

        db, img_name = os.path.split(self.image_src)
        file_name = db_path + "/" + img_name
        plt.savefig(file_name, dpi=500)
        plt.close()


