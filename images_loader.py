import glob
import cv2
import numpy


class ImagesLoader:
    def __init__(self, folder, batch_size):
        self.images = glob.glob('{}/*.png')
        self.batch_size = batch_size
        self.iters = ceil(len(self.images)/self.batch_size)
        self.pipe_line = []

    def __iter__(self):
        for i in range(self.iters):
            imgs = self.images[i*self.batch_size: min(len(self.images), (i+1)*self.batch_size)]
            imgs = [cv2.imread(img) for img in imgs]
            for fn in self.pipe_line:
                imgs = [fn(img) for img in imgs]
            imgs = np.stack(imgs)
            yield imgs

    def map(self, fn):
        pipe_line.append(fn)
