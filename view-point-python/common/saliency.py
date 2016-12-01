from skimage import io, img_as_float
from scipy import fftpack, ndimage, misc
from scipy.ndimage import uniform_filter, filters
from skimage.color import rgb2gray, rgb2lab
import numpy as np

import time
import msss_saliency as msss

class Saliency:
    def __init__(self, image, saliency_type = 0, filter_size=3, mode="nearest", sigma=2.5):
        # saliency_type
        self.timer = time.time()
        self.saliency_map = self.__find_saliency(image)

    def __find_saliency(self, image, size=5):
        gfrgb = filters.gaussian_filter(image, size)
         
        lab = rgb2lab(gfrgb)

        lab = np.ascontiguousarray(lab)

        saliency_map = msss.msss_saliency(lab)

        return saliency_map


    def __set_timer(self):
        self.timer = time.time()

    def __print_timer(self):
        print time.time() - self.timer

    def getSaliencyMap(self):
        return self.saliency_map
        
