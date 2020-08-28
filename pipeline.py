import cv2
import numpy as np

from config import *
from numba import prange

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.copyMakeBorder( img, 15, 15, 15, 15, cv2.BORDER_CONSTANT) # to get a better countour
    img = cv2.resize(img, tuple(IMAGE_SIZE))
    img = cv2.bilateralFilter(img, 5, 30, 30)
    return np.asarray(img, dtype='uint8')


def rotate(img):
    temp = cv2.GaussianBlur(img, (5,5), 0)
    _, temp = cv2.threshold(temp, 0.5*temp.max(),255,0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel, 1)
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            epsilon = 0.1*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
    src = []
    dst = [[0, 0], [0, 255], [255, 0], [255, 255]]

    for point in dst:
        min_dist = 50000
        min_p = None
        for p in approx:
            dist = abs(point-p).sum()
            if  dist < min_dist:
                min_dist = dist
                min_p = p
        src.append(min_p.tolist()[0])   
    src = np.float32(src); dst = np.float32(dst)
    
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img,M,tuple(IMAGE_SIZE))
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    return img


