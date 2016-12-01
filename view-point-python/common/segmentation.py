import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.segmentation import relabel_sequential
import time
import pylab as pl
from skimage import morphology, measure
import cv2
import cv
from scipy import ndimage
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import segmentation_local

MAX_NUM_SEGMENTS = 30000

# Fast segmentation
# min_dist = 0.17, compactness=10, n_segments=50, max_iter=10
class SuperPixelSegmentation:
    def __init__(self, image, seg_type=1, min_dist=0.008, compactness=10, n_segments=50, sigma=1.0, num_of_salient_objects=30, max_iter=100):
        self.timer = time.time()
        self.image = image
        self.min_dist = min_dist
        self.seg_type = seg_type
        self.num_of_salient_objects = num_of_salient_objects

        self.__set_timer("superpixel clustering")
        # self.slic_map, segments = slic(self.image, compactness=compactness, n_segments=n_segments, sigma=sigma, max_iter=max_iter)
        self.slic_map, segments = slic(self.image, compactness=compactness, n_segments=n_segments, sigma=sigma, max_iter=max_iter)
        print "Number of SLIC segments = ", len(np.unique(self.slic_map))
        self.__print_timer("superpixel clustering")

        self.__set_timer("merging superpixels")
        self.slic_map_0 = self.slic_map
        self.slic_map = self.__segmentation(segments, min_dist)
        self.__print_timer("merging superpixel")

        self.__set_timer("connected components")
        # dummy_map = morphology.label(self.slic_map, neighbors=4)
        dummy_map = measure.label(self.slic_map, connectivity=1)
        num_of_segments = len(np.unique(dummy_map))
        # pixel_count = self.__find_pixel_count(dummy_map, num_of_segments)
        print "Number of connected components = ", num_of_segments
        self.__print_timer("connected components")
        
        self.__set_timer("deleting small segments")
        self.min_num_pixel = self.__find_minimum_pixel_count(dummy_map)
        morphology.remove_small_objects(dummy_map, self.min_num_pixel, connectivity=4, in_place=True)
        self.__print_timer("deleting small segments")

        self.smap, self.fwd, self.inv = relabel_sequential(dummy_map, offset=1)
        print "Total number of segments remaining = ", len(np.unique(self.smap))

    def __set_timer(self, mesg=""):
        self.timer = time.time()
        if len(mesg) > 0: 
            print "Starting ", mesg, "..."

    def __print_timer(self, mesg=""):
        print mesg, "done. run time = ", time.time() - self.timer

    def __find_minimum_pixel_count(self, smap):
        # sorted_count =  sorted(pixel_count, reverse=True)
        # print sorted_count
        height, width = smap.shape
        min_num_pixel = height*width/200

        # if(sorted_count[self.num_of_salient_objects] < min_num_pixel):
        #     min_num_pixel = sorted_count[self.num_of_salient_objects] - 1

        return min_num_pixel

    def __delete_small_segments(self, smap):
        height, width = smap.shape
        for i in xrange(height):
            for j in xrange(width):
                if pixel_count[smap[i][j]] < self.min_num_pixel:
                    smap[i][j] = 0

        return smap

    def getDeletedSegmentID(self):
        return self.deleted_segment

    def getProminentColorSegment(self):
        return self.prom_color_segment
        
    def getSegmentationMap(self):
        return self.smap

    def getMeanColor(self):
        return self.mean_color, self.fwd, self.inv 

    def getSlicSegmentationMap(self):
        return self.slic_map_0

    def __segmentation(self, segments, min_dist):
        self.num_segments = np.amax(self.slic_map)+1
        #print self.num_segments
        #np.savetxt("slic_map.txt", self.slic_map, '%d')
        seg_map = self.slic_map

        # if need to merge small segments
        if self.seg_type == 1:
            self.mean_color = np.ascontiguousarray(segments[:,3:6])
            # self.mean_color = self.__find_mean_color_0()

            # edge_matrix_0 = self.__find_edges()
            
            optimize = True
            # calling cython code for this
            if optimize == True:
                seg_map = segmentation_local.merge_segments_wrap(self.mean_color, self.slic_map.astype(np.intp), min_dist)
            else:
                edge_matrix = self.__merge_edges(self.mean_color)
                seg_map = self.__merge_segments(edge_matrix)
        elif self.seg_type == 2:
            edge_matrix = self.__merge_edges2()
            seg_map = self.__merge_segments(edge_matrix)
        elif self.seg_type == 3:
            mean_vector = self.__extract_features()
            seg_map = self.__cluster_data(mean_vector)
            
        return seg_map

    def __extract_features(self):
        height, width, dim = self.image.shape
        mean_vector = np.zeros(shape=(self.num_segments, dim+2))
        pixel_count = np.zeros(self.num_segments)
        for i in xrange(height):
            for j in xrange(width):
                segment_ID = self.slic_map[i][j]
                r, g, b = self.image[i][j]
                pxlCnt = pixel_count[segment_ID]
                if pxlCnt == 0:
                    mean_vector[segment_ID] = r, g, b, i, j
                else:
                    mean_r, mean_g, mean_b, ii, jj = mean_vector[segment_ID]
                    mean_vector[segment_ID] = (mean_r + r), (mean_g + g), (mean_b + b), i + ii, j + jj
    
                pixel_count[segment_ID] = pxlCnt + 1
        
        for i in xrange(self.num_segments):
            mean_r, mean_g, mean_b, ii, jj = mean_vector[i]
            #print pixel_count[i], i
            num_of_pixel = pixel_count[i]
            mean_vector[i] = mean_r/num_of_pixel, mean_g/num_of_pixel, mean_b/num_of_pixel, ii/num_of_pixel, jj/num_of_pixel

        return mean_vector

    def __cluster_data(self, data):
        estimator = KMeans(init='k-means++', n_clusters=10, n_init=10)
        estimator.fit(data)

        slic_map2 = np.empty_like(self.slic_map)
        slic_map2[:] = self.slic_map

        height, width = self.slic_map.shape
        for i in xrange(height):
            for j in xrange(width):
                seg_number = self.slic_map[i][j]
                slic_map2[i][j] = estimator.labels_[seg_number]
        
        return slic_map2

    def __find_mean_color2(self):
        print "starting mean color..."
        timer = time.time()
        height, width, dim = self.image.shape
        mean_color = np.zeros(shape=(self.num_segments, dim), dtype=int)
        min_color = np.zeros(shape=(self.num_segments, dim), dtype=int)
        max_color = np.zeros(shape=(self.num_segments, dim), dtype=int)
        max_color[:] = np.inf
        #pixel_count = np.zeros(self.num_segments, dtype=int)
        for i in xrange(height):
            for j in xrange(width):
                segment_ID = self.slic_map[i][j]
                #mean_color[segment_ID] = self.image[i][j]
                #r, g, b = self.image[i][j]
                #pxlCnt = pixel_count[segment_ID]
                ##if pxlCnt == 0:
                ##    mean_color[segment_ID] = r, g, b
                ##else:
                #mean_r, mean_g, mean_b = mean_color[segment_ID]
                #mean_color[segment_ID] = (mean_r + r), (mean_g + g), (mean_b + b)
                #mean_color[segment_ID] += self.image[i][j]
                
                pixel_color = self.image[i][j]
                if (pixel_color > max_color[segment_ID]).all():
                    max_color[segment_ID] = pixel_color
                elif (pixel_color < min_color[segment_ID]).all():
                    min_color[segment_ID] = pixel_color

                #mean_color[segment_ID] += self.image[i][j]
                #pixel_count[segment_ID] += 1
    
        
        for i in xrange(self.num_segments):
            mean_color[i] = tuple([j/2 for j in max_color[i] + min_color[i]])
            #num_of_pixel = pixel_count[i]
            #mean_color[i] = tuple([j/pixel_count[i] for j in mean_color[i]])
            #mean_r, mean_g, mean_b = mean_color[i]
            #mean_color[i] = mean_r/num_of_pixel, mean_g/num_of_pixel, mean_b/num_of_pixel
        
        #max_color_seg = np.argmax(pixel_count)
        #print pixel_count
        #self.most_occured_color = mean_color[max_color_seg]
        #print self.most_occured_color

        print "mean color done. run time = ", time.time() - timer

        return mean_color

    
    def __find_mean_color_0(self):
        print "starting mean color..."
        timer = time.time()
        height, width, dim = self.image.shape
        mean_color = np.zeros(shape=(self.num_segments, dim))
        pixel_count = np.zeros(self.num_segments)
        for i in xrange(height):
            for j in xrange(width):
                segment_ID = self.slic_map[i][j]
                #mean_color[segment_ID] = self.image[i][j]
                #r, g, b = self.image[i][j]
                #pxlCnt = pixel_count[segment_ID]
                ##if pxlCnt == 0:
                ##    mean_color[segment_ID] = r, g, b
                ##else:
                #mean_r, mean_g, mean_b = mean_color[segment_ID]
                #mean_color[segment_ID] = (mean_r + r), (mean_g + g), (mean_b + b)
                #mean_color[segment_ID] += self.image[i][j]

                mean_color[segment_ID] += self.image[i][j]
                pixel_count[segment_ID] += 1
    
        
        for i in xrange(self.num_segments):
            #num_of_pixel = pixel_count[i]
            mean_color[i] = tuple([j/pixel_count[i] for j in mean_color[i]])
            #mean_r, mean_g, mean_b = mean_color[i]
            #mean_color[i] = mean_r/num_of_pixel, mean_g/num_of_pixel, mean_b/num_of_pixel
        
        #max_color_seg = np.argmax(pixel_count)
        #print pixel_count
        #self.most_occured_color = mean_color[max_color_seg]
        #print self.most_occured_color

        print "mean color done. run time = ", time.time() - timer

        return mean_color
    
    def __find_edges(self):
        height, width = self.slic_map.shape
        edge_matrix = np.zeros(shape=(self.num_segments, self.num_segments))
        for i in xrange(height-1):
            for j in xrange(width-1):
                c_segment = self.slic_map[i][j]
                right_segment = self.slic_map[i][j+1]
                bottom_segment = self.slic_map[i+1][j]
                if c_segment != right_segment:
                    edge_matrix[c_segment][right_segment] = 1
                    edge_matrix[right_segment][c_segment] = 1
                if c_segment != bottom_segment:
                    edge_matrix[c_segment][bottom_segment] = 1
                    edge_matrix[bottom_segment][c_segment] = 1
    
        return edge_matrix
    
    def __similar_segments(self, i, j, mean_color):
        a = mean_color[i]
        b = mean_color[j]
        # dist = np.abs(np.dot(a-b, a-b))
        dist = np.linalg.norm(a-b)
        # print dist
        #print a, b
        # print dist
        bSame = False
        if dist < self.min_dist:
            bSame = True
        return bSame
    
    def __similar_segments2(self, a, b):
        dist = np.linalg.norm(a-b)
        #print a, b
        print dist
        bSame = False
        if dist < self.min_dist:
            bSame = True
        return bSame


    def __merge_edges(self, mean_color):
        edge_matrix = np.zeros(shape=(self.num_segments, self.num_segments))
        height, width = edge_matrix.shape
        for i in xrange(height):
            for j in range(i, width):
                if edge_matrix[i][j] == 0:
                    if self.__similar_segments(i, j, mean_color) == True:
                        edge_matrix[i][j] = 2
                        edge_matrix[j][i] = 2
        #x1, y1 = 462, 237
        #x2, y2 = 433, 297
        #print mean_color[self.slic_map[x1][y1]]
        #print mean_color[self.slic_map[x2][y2]]
        #print np.linalg.norm(mean_color[self.slic_map[x1][y1]]-mean_color[self.slic_map[x2][y2]])
        #print self.slic_map[x1][y1], self.slic_map[x2][y2]
        #print edge_matrix[self.slic_map[x1][y1]][self.slic_map[x2][y2]]
        #print edge_matrix[self.slic_map[x2][y2]][self.slic_map[x1][y1]]

        return edge_matrix

    def __merge_edges2(self):
       height, width, dim = self.image.shape
       edge_matrix = np.zeros(shape=(self.num_segments, self.num_segments))
       
       for i in xrange(height-1):
           for j in xrange(width-1):
               c_segment = self.slic_map[i][j]
               right_segment = self.slic_map[i][j+1]
               bottom_segment = self.slic_map[i+1][j]
    
               if c_segment != right_segment:
                   if edge_matrix[c_segment][right_segment] == 0:
                       if self.__similar_segments2(self.image[i][j], self.image[i][j+1]) == True:
                           edge_matrix[c_segment][right_segment] = 2
                           edge_matrix[right_segment][c_segment] = 2
               if c_segment != bottom_segment:
                   if edge_matrix[c_segment][bottom_segment] == 0:
                       if self.__similar_segments2(self.image[i][j], self.image[i+1][j]) == True:
                           edge_matrix[c_segment][bottom_segment] = 2
                           edge_matrix[bottom_segment][c_segment] = 2
       return edge_matrix
    
    def __find_pixel_count(self, slic_map, num_of_segments):
        height, width = slic_map.shape
        pixel_count = np.zeros(num_of_segments)
        for i in xrange(height):
            for j in xrange(width):
                pixel_count[slic_map[i][j]] += 1
    
        return pixel_count

    def __find_connected_components(self, slic_map):
        height, width, dim = self.image.shape
        pixel_count = np.zeros(MAX_NUM_SEGMENTS)
        node_mark = np.zeros(shape=(height, width))
        slic_map2 = np.copy(slic_map)
        
        segment_id = 0
        stack = []
        for i in xrange(height-1):
            for j in xrange(width-1):                
                if node_mark[i][j] == 1:
                    continue
                node_mark[i][j] = 1
                pixel_count[segment_id] += 1
                slic_map2[i][j] = segment_id

                c_segment = slic_map[i][j]
                right_segment = -1
                bottom_segment = -1
                left_segment = -1
                top_segment = -1
                if j < width-1 and node_mark[i][j+1] == 0:
                    right_segment = slic_map[i][j+1]
                if j > 0 and node_mark[i][j-1] == 0:
                    left_segment = slic_map[i][j-1]
                if i < height-1 and node_mark[i+1][j] == 0:   
                    bottom_segment = slic_map[i+1][j]
                if i > 0 and node_mark[i-1][j] == 0:
                    top_segment = slic_map[i-1][j]

                if c_segment == right_segment:
                    stack.append((i, j+1))
                    pixel_count[segment_id] += 1
                    node_mark[i][j+1] = 1
                    slic_map2[i][j+1] = segment_id
            
                if c_segment == bottom_segment:
                    stack.append((i+1, j))
                    pixel_count[segment_id] += 1
                    node_mark[i+1][j] = 1
                    slic_map2[i+1][j] = segment_id

                if c_segment == left_segment:
                    stack.append((i, j-1))
                    pixel_count[segment_id] += 1
                    node_mark[i][j-1] = 1
                    slic_map2[i][j-1] = segment_id

                if c_segment == top_segment:
                    stack.append((i-1, j))
                    pixel_count[segment_id] += 1
                    node_mark[i-1][j] = 1
                    slic_map2[i-1][j] = segment_id

                while len(stack) > 0 :
                    ii, jj = stack.pop()

                    c_segment = slic_map[ii][jj]
                    right_segment = -1
                    bottom_segment = -1
                    left_segment = -1
                    top_segment = -1
                    if jj < width-1 and node_mark[ii][jj+1] == 0:
                        right_segment = slic_map[ii][jj+1]
                    if jj > 0 and node_mark[ii][jj-1] == 0:
                        left_segment = slic_map[ii][jj-1]
                    if ii < height-1 and node_mark[ii+1][jj] == 0:   
                        bottom_segment = slic_map[ii+1][jj]
                    if ii > 0 and node_mark[ii-1][jj] == 0:
                        top_segment = slic_map[ii-1][jj]

                    if c_segment == right_segment:
                        stack.append((ii, jj+1))
                        pixel_count[segment_id] += 1
                        node_mark[ii][jj+1] = 1
                        slic_map2[ii][jj+1] = segment_id
            
                    if c_segment == bottom_segment:
                        stack.append((ii+1, jj))
                        pixel_count[segment_id] += 1
                        node_mark[ii+1][jj] = 1
                        slic_map2[ii+1][jj] = segment_id

                    if c_segment == left_segment:
                        stack.append((ii, jj-1))
                        pixel_count[segment_id] += 1
                        node_mark[ii][jj-1] = 1
                        slic_map2[ii][jj-1] = segment_id

                    if c_segment == top_segment:
                        stack.append((ii-1, jj))
                        pixel_count[segment_id] += 1
                        node_mark[ii-1][jj] = 1
                        slic_map2[ii-1][jj] = segment_id

                segment_id+=1


        return slic_map2, pixel_count

    def __merge_segments(self, edge_matrix):
        height, width = edge_matrix.shape
        slic_map2 = np.empty_like(self.slic_map)
        segment_label = np.zeros(self.num_segments)
        node_mark = np.zeros(self.num_segments)
        slic_map2[:] = self.slic_map
        for i in xrange(height):
            segment_label[i] = i   # mark the segment number 
        
        stack = []
        for i in xrange(height):
            if node_mark[i] == 1:
                continue
            node_mark[i] = 1

            # Add all the connected node to a stack for traversal
            for j in range(i+1, width):                
                if edge_matrix[i][j] == 2 and node_mark[j] == 0:
                    stack.append(j)
                    segment_label[j] = segment_label[i]
                    node_mark[j] = 1
            
            while len(stack) > 0 :
                node = stack.pop()
                for k in xrange(width):
                    if edge_matrix[node][k] == 2 and node_mark[k] == 0:
                        stack.append(k)
                        segment_label[k] = segment_label[i]
                        node_mark[k] = 1
    
        height, width = self.slic_map.shape
        for i in xrange(height):
            for j in xrange(width):
                seg_number = self.slic_map[i][j]
                slic_map2[i][j] = segment_label[seg_number]
        #print segment_label[125]
        #print segment_label[68]
        return slic_map2

