#!python
#cython: boundscheck=False
#cython: cdivision=True
#cython: cdivision_warnings=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: overflowcheck=False

import numpy as np
cimport numpy as np
import time

ctypedef np.int_t DTYPE_t

def merge_segments_wrap(double[:,::1] mean_color, Py_ssize_t[:, :] slic_map, double min_dist):
    cdef Py_ssize_t num_segments = np.amax(slic_map)+1

    edge_matrix = merge_edges(mean_color, num_segments, min_dist)
    seg_map = merge_segments(slic_map, num_segments, edge_matrix)
        
    return seg_map


cdef merge_edges(double[:,::1] mean_color, Py_ssize_t num_segments, double min_dist):
    cdef Py_ssize_t[:,:] edge_matrix = np.zeros(shape=(num_segments, num_segments), dtype=np.intp)
    cdef double dist_color

    cdef Py_ssize_t i, j, c
    for i in range(num_segments):
        for j in range(i, num_segments):
            if edge_matrix[i][j] == 0:
                dist_color = 0.0
                for c in range(3):
                    dist_color += (mean_color[i][c] - mean_color[j][c]) ** 4

                # print "%.6f" % dist_color
                if dist_color < min_dist :
                    edge_matrix[i][j] = 2
                    edge_matrix[j][i] = 2

    return edge_matrix


cdef merge_segments(Py_ssize_t[:, :] slic_map, Py_ssize_t num_segments, Py_ssize_t[:, :] edge_matrix):
    cdef Py_ssize_t height, width
    height = <Py_ssize_t>slic_map.shape[0]
    width = <Py_ssize_t>slic_map.shape[1]

    cdef Py_ssize_t[:, :] new_map = np.zeros(shape=(height, width), dtype=np.intp)
    new_map = slic_map

    cdef Py_ssize_t[:, :] seg_map = np.empty((height, width), dtype=np.intp)
    cdef Py_ssize_t[:] segment_label = np.zeros(num_segments, dtype=np.intp)
    cdef Py_ssize_t[:] node_mark = np.zeros(num_segments, dtype=np.intp)

    cdef Py_ssize_t i, j, k, seg_number

    for i in range(num_segments):
        segment_label[i] = i   # mark the segment number 
    
    stack = []
    for i in range(num_segments):
        if node_mark[i] == 1:
            continue
        node_mark[i] = 1

        # Add all the connected node to a stack for traversal
        for j in range(i+1, num_segments):                
            if edge_matrix[i][j] == 2 and node_mark[j] == 0:
                stack.append(j)
                segment_label[j] = segment_label[i]
                node_mark[j] = 1
        
        while len(stack) > 0 :
            node = stack.pop()
            for k in xrange(num_segments):
                if edge_matrix[node][k] == 2 and node_mark[k] == 0:
                    stack.append(k)
                    segment_label[k] = segment_label[i]
                    node_mark[k] = 1

    for i in range(height):
        for j in range(width):
            seg_number = <Py_ssize_t>new_map[i][j]
            seg_map[i][j] = segment_label[seg_number]

    slic_map2 = np.asarray(seg_map)

    return slic_map2

