#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from math import pow
import numpy as np
cimport numpy as np


cdef double iisum(double[:, :] iimg, Py_ssize_t x1, Py_ssize_t y1, Py_ssize_t x2, Py_ssize_t y2):
    cdef double sum
    sum = 0
    if(x1>1 and y1>1) :
        sum = iimg[y2-1,x2-1]+iimg[y1-2,x1-2]-iimg[y1-2,x2-1]-iimg[y2-1,x1-2]
    elif(x1<=1 and y1>1) :
        sum = iimg[y2-1,x2-1]-iimg[y1-2,x2-1]
    elif(y1<=1 and x1>1) :
        sum = iimg[y2-1,x2-1]-iimg[y2-1,x1-2]
    else :
        sum = iimg[y2-1,x2-1]
    
    return sum


def msss_saliency(double[:, :, ::1] lab):
    
    cdef Py_ssize_t height, width, dim
    height = lab.shape[0]
    width = lab.shape[1]
    dim = lab.shape[2]
    
    cdef double[:, :] l = lab[:,:,0]
    # lm = np.mean(l)
    
    cdef double[:, :] a = lab[:,:,1]
    # am = np.mean(a)
    
    cdef double[:, :] b = lab[:,:,2]
    # bm = np.mean(b)
    
    # create integral images
    cdef double[:, :] li = np.cumsum(np.cumsum(l, axis=1), axis=0)
    cdef double[:, :] ai = np.cumsum(np.cumsum(a, axis=1), axis=0)
    cdef double[:, :] bi = np.cumsum(np.cumsum(b, axis=1), axis=0)

    cdef double[:, :] sm = np.empty(shape=(height, width), dtype=np.double)

    cdef double invarea, lm, am, bm

    cdef Py_ssize_t j, k, yo, y1, y2, xo, x1, x2
    for j in range(1, height+1):
        yo = <Py_ssize_t>min(j, height-j)
        y1 = <Py_ssize_t>max(1,j-yo)
        y2 = <Py_ssize_t>min(j+yo,height)
        for k in range(1, width+1):
            xo = <Py_ssize_t>min(k,width-k)
            x1 = <Py_ssize_t>max(1,k-xo)
            x2 = <Py_ssize_t>min(k+xo,width)
            invarea = 1.0/((y2-y1+1)*(x2-x1+1))
            lm = iisum(li,x1,y1,x2,y2)*invarea
            am = iisum(ai,x1,y1,x2,y2)*invarea
            bm = iisum(bi,x1,y1,x2,y2)*invarea
            #---------------------------------------------------------
            # Compute the saliency map
            #---------------------------------------------------------
            sm[j-1,k-1] = (l[j-1,k-1]-lm)**2 + (a[j-1,k-1]-am)**2 + (b[j-1,k-1]-bm)**2

    img = (sm-np.min(sm))/(np.max(sm)-np.min(sm))

    return img


