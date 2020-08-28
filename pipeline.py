import cv2
import numpy as np

from config import *
from numba import prange


def _sort_src_from_dst(src, dst):
    ret = []

    for point in dst:
        min_dist = 50000
        min_p = None
        for p in src:
            dist = abs(point-p).sum()
            if  dist < min_dist:
                min_dist = dist
                min_p = p
        ret.append(min_p.tolist())   
    ret = np.float32(ret); dst = np.float32(dst)
    return ret, dst

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

    dst = [[0, 0], [0, 255], [255, 0], [255, 255]]
    src, dst = _sort_src_from_dst(approx, dst)
    
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img,M,tuple(IMAGE_SIZE))
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    return img


def boxes_divide(img):
    boxes = np.zeros((81, *BOX_SIZE), dtype='uint8')
    for i in prange(81):
        rows = IMAGE_SIZE[0]//9; cols = IMAGE_SIZE[1]//9
        r = i//9
        c = i%9
        margin = 2
        box = img[max(0, r*rows - margin): min(IMAGE_SIZE[0], (r+1)*rows+ margin), \
            max(0, c*cols-margin): min(IMAGE_SIZE[1], (c+1)*cols+margin)]
        box = cv2.resize(box, tuple(BOX_SIZE))
        boxes[i] = box
    return boxes

def boxes_refine(boxes):
    for i in prange(81):
        box = boxes[i]
        contours, hierarchy = cv2.findContours(box, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                rect = cv2.minAreaRect(contour)
                rect = cv2.boxPoints(rect)

        dst = [[0, 0], [0, 28], [28, 0], [28, 28]]
        src, dst = _sort_src_from_dst(rect, dst)

        M = cv2.getPerspectiveTransform(src, dst)
        box = cv2.warpPerspective(box, M, tuple(BOX_SIZE))
        box = cv2.adaptiveThreshold(box, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, 11, 2)

        boxes[i] = box
    return boxes