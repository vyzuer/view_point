#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: cdivision_warnings=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: overflowcheck=False

import numpy as np
cimport numpy as np

import cv2
import time
import threading
from skimage.color import rgb2gray, rgb2hsv

from sklearn.decomposition import PCA
from sklearn import preprocessing

ctypedef np.int32_t dtypei_t

init_pos = np.array([[1.0/6, 1.0/3], [1.0/6, 2.0/3], [1.0/3, 1.0/6], [1.0/3, 1.0/3], [1.0/3, 1.0/2], [1.0/3, 2.0/3], [1.0/3, 5.0/6], [1.0/2, 1.0/3], [1.0/2, 1.0/2], [1.0/2, 2.0/3], [2.0/3, 1.0/6], [2.0/3, 1.0/3], [2.0/3, 1.0/2], [2.0/3, 2.0/3], [2.0/3, 5.0/6], [5.0/6, 1.0/3], [5.0/6, 2.0/3]])

# init_pos = np.array([[1.0/3, 1.0/3], [1.0/3, 2.0/3], [1.0/2, 1.0/2], [2.0/3, 1.0/3], [2.0/3, 2.0/3]])

def detect_saliency_of_segments(Py_ssize_t[:, :] seg_map, double[:, :] sal_map, double min_saliency):
    cdef Py_ssize_t num_segments = np.amax(seg_map)+1

    cdef Py_ssize_t height, width
    height = <Py_ssize_t>seg_map.shape[0]
    width = <Py_ssize_t>seg_map.shape[1]

    cdef double[:] saliency_list = np.zeros(num_segments+1, dtype=np.double)
    cdef Py_ssize_t[:] pixel_count = np.zeros(num_segments, dtype=np.intp)

    cdef Py_ssize_t[:, :] new_map = np.zeros(shape=(height, width), dtype=np.intp)

    new_map = seg_map

    cdef Py_ssize_t i, j, seg_id

    for i in xrange(height):
        for j in xrange(width):
            if new_map[i][j] != 0:
                seg_id = <Py_ssize_t>new_map[i][j]
                saliency_list[seg_id] += sal_map[i][j]
                pixel_count[seg_id] += 1

    for i in xrange(num_segments):
        saliency_list[i] = saliency_list[i]/(pixel_count[i]+1)

    # saliency_list = saliency_list/max(saliency_list)
    cdef Py_ssize_t[:] salient_objects = np.ones(num_segments, dtype=np.intp)
    for i in xrange(num_segments):
        if saliency_list[i] < min_saliency :
            salient_objects[i] = 0

    for i in xrange(height):
        for j in xrange(width):
            seg_id = <Py_ssize_t>new_map[i][j]
            if salient_objects[seg_id] == 0:
                new_map[i][j] = 0
                

    sal_list = np.asarray(saliency_list)
    sal_list = sal_list/sal_list.max()

    # face saliency
    sal_list[num_segments] = 1.0

    return sal_list, np.asarray(salient_objects), np.asarray(pixel_count), np.asarray(new_map)


def comp_map(Py_ssize_t grid_height, Py_ssize_t grid_width, Py_ssize_t block_height, Py_ssize_t block_width, Py_ssize_t[:, :] seg_map, double[:] saliency_list):

    cdef Py_ssize_t i, j, k, l, m, seg_id, f_map_height, f_map_width
    cdef Py_ssize_t img_height, img_width
    f_map_height = grid_height*block_height
    f_map_width = grid_width*block_width

    img_height = seg_map.shape[0]
    img_width = seg_map.shape[1]

    cdef Py_ssize_t[:, :] new_map = np.zeros(shape=(img_height, img_width), dtype=np.intp)
    new_map = seg_map

    cdef Py_ssize_t num_segments = np.amax(seg_map)+1

    cdef double[:] new_saliency_list = np.zeros(num_segments, dtype=np.double)
    new_saliency_list = saliency_list

    cdef double[:, :] feature_map = np.zeros(shape=(f_map_height, f_map_width), dtype=np.double)
    
    cdef double x_step, y_step, t_step, bx_step, by_step
    cdef double pix_x, pix_y, block_x, block_y, saliency_norm
    # cell size for a image
    x_step = 1.0*img_height/f_map_height
    y_step = 1.0*img_width/f_map_width
    t_step = x_step*y_step
    # block steps
    bx_step = 1.0*img_height/grid_height
    by_step = 1.0*img_width/grid_width

    # current location of our spanning
    pix_x = 0.0
    pix_y = 0.0
    block_x = 0.0
    block_y = 0.0

    cdef Py_ssize_t x_0, y_0, ii, jj
    for i in range(grid_height):
        pix_x = block_x
        block_y = 0.0
        for j in range(grid_width):
            pix_x = block_x
            pix_y = block_y
            for k in range(block_height):
                x_0 = <Py_ssize_t>pix_x
                pix_x += x_step

                pix_y = block_y
                for l in range(block_width):
                    y_0 = <Py_ssize_t>pix_y
                    pix_y += y_step
                    saliency_sum = 0.0
                    for m in range(x_0, <Py_ssize_t>pix_x):
                        for n in range(y_0, <Py_ssize_t>pix_y):
                            # print i, j, k, l, m, n
                            seg_id = new_map[m][n]
                            saliency_sum += new_saliency_list[seg_id]

                    saliency_norm = saliency_sum/t_step
                    # saliency_norm = saliency_sum/((pix_x - x_0)*(pix_y - y_0))
                    # print saliency_norm
                    ii = i*block_height + k
                    jj = j*block_width + l
                    feature_map[ii][jj] = saliency_norm
            
            block_y += by_step
        
        block_x += bx_step

    return np.asarray(feature_map)

def find_comp_vec_2(double[:, :] seg_map, Py_ssize_t n_xcells, Py_ssize_t n_ycells, Py_ssize_t n_bxcells, Py_ssize_t n_bycells, Py_ssize_t nbins):

    cdef double[:] bins = np.linspace(0, 1, nbins+1)

    cdef Py_ssize_t img_height, img_width
    img_height= seg_map.shape[0]
    img_width = seg_map.shape[1]

    cdef double[:, :] new_saliency_map = np.zeros(shape=(img_height, img_width), dtype=np.double)
    # cdef double[:, :] saliency_block
    new_saliency_map = seg_map

    cdef Py_ssize_t i, j
    cdef double x_step, y_step, bx_step, by_step
    # cell size for a image
    x_step = 1.0*img_height/n_xcells
    y_step = 1.0*img_width/n_ycells

    # block steps
    bx_step = x_step*n_bxcells
    by_step = y_step*n_bycells

    cdef Py_ssize_t n_xblocks, n_yblocks, n_dims
    n_xblocks = <Py_ssize_t>(n_xcells - n_bxcells + 1)
    n_yblocks = <Py_ssize_t>(n_ycells - n_bycells + 1)

    n_dims = <Py_ssize_t>(nbins*n_xblocks*n_yblocks)

    cdef Py_ssize_t[:] feature_vector = np.zeros(n_dims, dtype=np.intp)
    cdef Py_ssize_t[:] hist = np.zeros(nbins+1, dtype=np.intp)

    cdef double x_pos, y_pos
    cdef Py_ssize_t count, x_min, y_min, x_max, y_max
    x_pos = 0.0
    y_pos = 0.0
    count = 0
    for i in range(n_xblocks):
        # reset y_pos to 0
        y_pos = 0.0
        for j in range(n_yblocks):
            x_min = <Py_ssize_t>x_pos
            y_min = <Py_ssize_t>y_pos
            x_max = <Py_ssize_t>(x_pos + bx_step)
            y_max = <Py_ssize_t>(y_pos + by_step)

            # extract histogram for this block
            saliency_block = np.asarray(new_saliency_map[x_min:x_max, y_min:y_max], dtype=np.double)
            # hist, bin_edges = np.histogram(saliency_block, nbins, (0,1))
            # hist = histogram(saliency_block, saliency_block, nbins, (0,1))[3]
            # print hist

            digitized = np.digitize(saliency_block.ravel(), bins, right=True)
            hist = np.bincount(digitized, minlength=nbins+1)
            # print hist[1:nbins+1]

            feature_vector[<Py_ssize_t>count:<Py_ssize_t>(count+nbins)] = hist[1:<Py_ssize_t>(nbins+1)]
            count += nbins

            # increment by one cell step
            y_pos = (j+1)*y_step

        # increment x_step by one cell
        x_pos = (i+1)*x_step

    return np.asarray(feature_vector, dtype=np.double)

# modify find_comp_vec_2 as well if you change this function
def find_comp_vec(Py_ssize_t[:, :] seg_map, double[:] saliency_list, Py_ssize_t n_xcells, Py_ssize_t n_ycells, Py_ssize_t n_bxcells, Py_ssize_t n_bycells, Py_ssize_t nbins):

    cdef double[:] bins = np.linspace(0, 1, nbins+1)

    cdef Py_ssize_t img_height, img_width
    img_height= seg_map.shape[0]
    img_width = seg_map.shape[1]

    cdef Py_ssize_t num_segments = np.amax(seg_map)+1

    cdef double[:, :] new_saliency_map = np.zeros(shape=(img_height, img_width), dtype=np.double)
    # cdef double[:, :] saliency_block
    cdef double[:] new_saliency_list = np.zeros(num_segments, dtype=np.double)
    new_saliency_list = saliency_list

    cdef Py_ssize_t[:, :] new_map = np.zeros(shape=(img_height, img_width), dtype=np.intp)
    new_map = seg_map

    cdef Py_ssize_t i, j, seg_id
    for i in range(img_height):
        for j in range(img_width):
            seg_id = new_map[i][j]
            new_saliency_map[i][j] = new_saliency_list[seg_id]

    cdef double x_step, y_step, bx_step, by_step
    # cell size for a image
    x_step = 1.0*img_height/n_xcells
    y_step = 1.0*img_width/n_ycells

    # block steps
    bx_step = x_step*n_bxcells
    by_step = y_step*n_bycells

    cdef Py_ssize_t n_xblocks, n_yblocks, n_dims
    n_xblocks = <Py_ssize_t>(n_xcells - n_bxcells + 1)
    n_yblocks = <Py_ssize_t>(n_ycells - n_bycells + 1)

    n_dims = <Py_ssize_t>(nbins*n_xblocks*n_yblocks)

    cdef Py_ssize_t[:] feature_vector = np.zeros(n_dims, dtype=np.intp)
    cdef Py_ssize_t[:] hist = np.zeros(nbins, dtype=np.intp)
    # feature_vector = np.zeros(n_dims)

    cdef double x_pos, y_pos
    cdef Py_ssize_t count, x_min, y_min, x_max, y_max
    x_pos = 0.0
    y_pos = 0.0
    count = 0
    for i in range(n_xblocks):
        # reset y_pos to 0
        y_pos = 0.0
        for j in range(n_yblocks):
            x_min = <Py_ssize_t>x_pos
            y_min = <Py_ssize_t>y_pos
            x_max = <Py_ssize_t>(x_pos + bx_step)
            y_max = <Py_ssize_t>(y_pos + by_step)

            # extract histogram for this block
            saliency_block = np.asarray(new_saliency_map[x_min:x_max, y_min:y_max], dtype=np.double)
            # saliency_block = new_saliency_map[x_min:x_max, y_min:y_max]
            # hist, bin_edges = np.histogram(saliency_block, nbins, (0,1))

            digitized = np.digitize(saliency_block.ravel(), bins, right=True)
            hist = np.bincount(digitized, minlength=nbins+1)
            # print hist[1:nbins+1]

            feature_vector[<Py_ssize_t>count:<Py_ssize_t>(count+nbins)] = hist[1:<Py_ssize_t>(nbins+1)]

            # feature_vector[<Py_ssize_t>count:<Py_ssize_t>(count+nbins)] = hist
            count += nbins

            # increment by one cell step
            y_pos = (j+1)*y_step

        # increment x_step by one cell
        x_pos = (i+1)*x_step

    return np.asarray(feature_vector, dtype=np.double), new_saliency_map
    

def find_view_context(unsigned char[:, :, ::1] image, Py_ssize_t n_xcells, Py_ssize_t n_ycells, Py_ssize_t n_bxcells, Py_ssize_t n_bycells, Py_ssize_t nbins):

    cdef Py_ssize_t img_height, img_width, img_dim
    # cell size for a image
    img_height= image.shape[0]
    img_width = image.shape[1]
    img_dim = image.shape[2]


    cdef double x_step, y_step, bx_step, by_step
    x_step = 1.0*img_height/n_xcells
    y_step = 1.0*img_width/n_ycells

    # block steps
    bx_step = x_step*n_bxcells
    by_step = y_step*n_bycells
    
    cdef Py_ssize_t n_xblocks, n_yblocks, n_dims
    n_xblocks = <Py_ssize_t>(n_xcells - n_bxcells + 1)
    n_yblocks = <Py_ssize_t>(n_ycells - n_bycells + 1)

    n_dims = <Py_ssize_t>(3*nbins*n_xblocks*n_yblocks)

    cdef Py_ssize_t[:] feature_vector = np.zeros(n_dims, dtype=np.intp)
    cdef Py_ssize_t[:] hist1 = np.zeros(nbins, dtype=np.intp)
    cdef Py_ssize_t[:] hist2 = np.zeros(nbins, dtype=np.intp)
    cdef Py_ssize_t[:] hist3 = np.zeros(nbins, dtype=np.intp)
    # feature_vector = np.zeros(n_dims)

    cdef double x_pos, y_pos
    cdef Py_ssize_t count, x_min, y_min, x_max, y_max

    cdef unsigned char[:, :, ::1] image_new = np.zeros(shape=(img_height, img_width, img_dim), dtype=np.ubyte)
    image_new = image
    # cdef double[:, :, ::1] img_hsv = np.zeros(shape=(img_height, img_width, img_dim), dtype=np.double)
    # img_hsv = rgb2hsv(image)

    x_pos = 0.0
    y_pos = 0.0
    count = 0
    cdef Py_ssize_t i, j
    for i in range(n_xblocks):
        # reset y_pos to 0
        y_pos = 0.0
        for j in range(n_yblocks):
            x_min = <Py_ssize_t>(x_pos)
            y_min = <Py_ssize_t>(y_pos)
            x_max = <Py_ssize_t>(x_pos + bx_step)
            y_max = <Py_ssize_t>(y_pos + by_step)

            # extract color histogram for this block
            # img_block = img_hsv[x_min:x_max, y_min:y_max,:]
            img_block = image[x_min:x_max, y_min:y_max,:]

            h, s, v = cv2.split(np.asarray(img_block))

            hist1, bin_edges1 = np.histogram(h, nbins, (0,256))
            hist2, bin_edges2 = np.histogram(s, nbins, (0,256))
            hist3, bin_edges3 = np.histogram(v, nbins, (0,256))

            feature_vector[<Py_ssize_t>count:<Py_ssize_t>(count+nbins)] = hist1
            count += nbins
            feature_vector[<Py_ssize_t>count:<Py_ssize_t>(count+nbins)] = hist2
            count += nbins
            feature_vector[<Py_ssize_t>count:<Py_ssize_t>(count+nbins)] = hist3
            count += nbins

            # increment by one cell step
            y_pos = (j+1)*y_step

        # increment x_step by one cell
        x_pos = (i+1)*x_step

    new_f_vec = np.asarray(feature_vector, dtype=np.double)
    cv2.normalize(new_f_vec, new_f_vec, 0, 1, cv2.NORM_MINMAX)

    return new_f_vec


cdef modify_map(double[:, :] new_map, faces, current_pos):
    
    cdef Py_ssize_t idx1, idx2
    cdef Py_ssize_t img_height, img_width
    img_height= new_map.shape[0]
    img_width = new_map.shape[1]

    cdef double[:, :] seg_map2 = np.zeros(shape=(img_height, img_width), dtype=np.double)
    seg_map2 = new_map

    faces_ = (faces*current_pos[2]/100).astype(np.int)
    # print faces_
    for (x, y, w, h) in faces_:
        # print x, y, w, h
        # x - horizontal position
        # y - vertical position
        # w - width
        # h - height
        
        for i in range(w):
            for j in range(h):
                try:
                    idx1 = <Py_ssize_t>(current_pos[0]+y+j)
                    idx2 = <Py_ssize_t>(current_pos[1]+x+i)
                    # seg_map2[current_pos[0]+y+j][current_pos[1]+x+i] = 1.0
                    seg_map2[idx1][idx2] = 1.0
                except IndexError:
                    return seg_map2, False

    return seg_map2, True


def find_neighbors(current_pos, y_step, x_step, z_step):
    a = current_pos[0]-y_step, current_pos[1], current_pos[2]
    b = current_pos[0], current_pos[1]+x_step, current_pos[2]
    c = current_pos[0]+y_step, current_pos[1], current_pos[2]
    d = current_pos[0], current_pos[1]-x_step, current_pos[2]
    e = current_pos[0], current_pos[1], current_pos[2]-z_step
    f = current_pos[0], current_pos[1], current_pos[2]+z_step

    n_list = np.array([a, b, c, d, e, f])
    # print n_list

    return n_list


def worker(pos, new_map, faceInfo, view_c, scalar, pca, regressor, img_height, img_width, box_h, box_w, y_step, x_step, z_step, thread_id, best_pos_list, best_score_list):
    best_score = 0.0
    best_pos = pos
    current_pos = pos
    change = True

    # timer = time.time()
    modified_map, valid = modify_map(new_map, faceInfo, current_pos)
    # print "run time modified map = ", time.time() - timer
    # print modified_map.shape
    # timer = time.time()
    fv = find_comp_vec_2(modified_map, 9, 9, 3, 3, 20)
    cv2.normalize(fv, fv, 0, 1, cv2.NORM_MINMAX)
    # print "run time comsposition 1 = ", time.time() - timer

    # form feature
    X = []
    X.extend(fv)
    X.extend(view_c)
    # X.extend(geo)

    X = np.asarray(X)
    # print X.shape
    X = np.reshape(X, (1, -1))
    # print X.shape

    X = scalar.transform(X)
    X = preprocessing.normalize(X, norm='l2')
    X = pca.transform(X)
    score = regressor.predict_proba(X)

    # change = False
    while change == True:
        change = False

        # check boundary conditions
        if current_pos[0] + box_h > img_height:
            break

        if current_pos[0] - box_h < 0:
            break

        if current_pos[1] + box_w > img_width:
            break

        if current_pos[1] - box_w < 0:
            break

        if current_pos[2] > 120:
            break

        # traverse neighbors of current point
        neighbors_list = find_neighbors(current_pos, y_step, x_step, z_step)
        for p in (neighbors_list):
            modified_map, valid = modify_map(new_map, faceInfo, p)
            if valid == False:
                change = False
                break

            # timer = time.time()
            fv = find_comp_vec_2(modified_map, 9, 9, 3, 3, 20)
            cv2.normalize(fv, fv, 0, 1, cv2.NORM_MINMAX)
            # fv = find_feature_vector(modified_map)
            # print "run time = ", time.time() - timer

            # form feature
            X = []
            X.extend(fv)
            X.extend(view_c)
            # X.extend(geo)

            X = np.asarray(X)
            # print X.shape
            X = np.reshape(X, (1, -1))
            # print X.shape

            X = scalar.transform(X)
            X = preprocessing.normalize(X, norm='l2')
            X = pca.transform(X)
            score_local = regressor.predict_proba(X)

            if score_local[0][0] > score[0][0] :
                score = score_local
                current_pos = p
                change = True

    if best_score < score[0][0]:
        best_score = score[0][0]
        best_pos = current_pos

    best_pos_list[thread_id] = best_pos
    best_score_list[thread_id] = best_score
    

def find_best_position(double[:, :] new_map, faceInfo, double[:] view_c, regressor, scalar, pca):
    img_height= new_map.shape[0]
    img_width = new_map.shape[1]
    # find bounding box for faces
    min_x, min_y = 10000, 10000
    max_x, max_y = 0, 0
    for (x, y, w, h) in faceInfo:
        # x - horizontal position
        # y - vertical position
        # w - width
        # h - height
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y

        if max_x < x+w:
            max_x = x+w
        if max_y < y+h:
            max_y = y+h

    # print min_x, min_y, max_x, max_y
    box_h = max_y - min_y
    box_w = max_x - min_x
    # print box_h, box_w

    cdef Py_ssize_t i, j, idx1, idx2
    # delete faces first
    for (x, y, w, h) in faceInfo:
            # x - horizontal position
            # y - vertical position
            # w - width
            # h - height
            for i in range(w):
                for j in range(h):
                    idx1 = <Py_ssize_t>(y+j)
                    idx2 = <Py_ssize_t>(x+i)
                    new_map[idx1][idx2] = 0.0

    x_step = int(img_width/50)
    y_step = int(img_height/50)
    # z_step = int((x_step+y_step)/2)
    z_step = 5


    x_init = (min_x+max_x)/2
    y_init = (min_y+max_y)/2

    pos_iter = init_pos*[img_height, img_width]
    pos_iter = np.insert(pos_iter, 0, [y_init, x_init], axis=0).astype(np.int)
    pos_iter = np.insert(pos_iter, 2, 100, axis=1)

    # plt.scatter(pos_iter[:, 1], pos_iter[:, 0])
    # plt.xlim(0, img_width)
    # plt.ylim(0, img_height)
    # plt.show()

    # -----> x direction
    # |
    # |
    # V
    # y direction
    # print pos_iter


    # move the origin to center of boundig box
    # print faceInfo
    orig_pos = [y_init, x_init, 0]
    faceInfo = faceInfo-[x_init, y_init, 0, 0]
    # print faceInfo
    best_score = 0.0
    best_pos = y_init, x_init, 100
    threads = []
    num_threads = len(pos_iter)
    best_pos_list = np.zeros(shape=(num_threads, 3), dtype=np.int)
    best_score_list = np.zeros(num_threads)
    thread_id = 0
    for pos in (pos_iter):
        t = threading.Thread(target=worker, args=(pos, new_map, faceInfo, view_c, scalar, pca, regressor, img_height, img_width, box_h, box_w, y_step, x_step, z_step, thread_id, best_pos_list, best_score_list, ))
        threads.append(t)
        t.start()
        thread_id += 1

    for t in threads:
        t.join()

    for i in range(num_threads):
        if best_score < best_score_list[i]:
            best_score = best_score_list[i]
            best_pos = best_pos_list[i]

    return best_pos, best_score, faceInfo, orig_pos

