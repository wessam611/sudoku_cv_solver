import glob
import cv2
import numpy
import math

from multiprocessing import Pool

import numpy as np

class ImagesLoader:
    def __init__(self, folder, batch_size):
        self.images = glob.glob('{}/*.png'.format(folder))
        self.batch_size = batch_size
        self.iters = math.ceil(len(self.images)/self.batch_size)
        self.pipeline = []

    def __iter__(self):
        for i in range(self.iters):
            pool = Pool()
            imgs = self.images[i*self.batch_size: min(len(self.images), (i+1)*self.batch_size)]
            imgs = pool.map(cv2.imread, imgs)
            for fn in self.pipeline:
                imgs = pool.map(fn, imgs)
            imgs = np.stack(imgs)
            yield imgs

    def map(self, fn):
        self.pipeline.append(fn)
        return self
