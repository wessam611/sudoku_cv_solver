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
    b = 5 # border
    dst = [[b, b], [b, 255-b], [255-b, b], [255-b, 255-b]]
    src, dst = _sort_src_from_dst(approx, dst)
    
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img,M,tuple(IMAGE_SIZE))
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

def remove_grid_lines(img):
    horizontal = 255 - np.copy(img)
    vertical = 255 - np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = cols // 15
    vertical_size = cols // 15
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = np.asarray(vertical>=127, dtype='bool')
    horizontal = np.asarray(horizontal>=127, dtype='bool')

    grid = np.asarray(vertical | horizontal, dtype='uint8')*255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    grid = 255-cv2.dilate(grid, kernel)

    grid = np.asarray(grid>=127, dtype='bool')
    img = np.asarray(img<=127, dtype='bool')
    img = np.asarray(img & grid, dtype='uint8')*255
    return img


def boxes_divide(img):
    boxes = np.zeros((81, *BOX_SIZE), dtype='uint8')
    b = 0 # border
    for i in prange(81):
        rows = (IMAGE_SIZE[0]-b)//9; cols = (IMAGE_SIZE[1]-b)//9
        r = i//9
        c = i%9
        margin = 4
        box = img[b+max(0, r*rows - margin): b+min(IMAGE_SIZE[0], (r+1)*rows+ margin), \
            b+max(0, c*cols-margin): b+min(IMAGE_SIZE[1], (c+1)*cols+margin)]
        box = cv2.resize(box, tuple(BOX_SIZE))
        boxes[i] = box
    return boxes

def boxes_refine(boxes):
    for i in prange(81):
        box = boxes[i]
        br = 10 # border
        th = 5
        if np.sum(box[br:-br, br:-br]) <= 255*th:
            box = np.zeros(BOX_SIZE, dtype='uint8')
            boxes[i] = box
            continue
        contours, hierarchy = cv2.findContours(box, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        rect = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                x,y,w,h = cv2.boundingRect(contour)
                if w >= h:
                    h = w
                    y -= w//2
                else:
                    w = h
                    x -= h//2
                rect = np.asarray([[x, y], [x+w, y], [x, y+h], [x+h, y+h]])
        if rect is None:
            continue
        sh = -8
        dst = [[0-sh, 0-sh], [0-sh, 28+sh], [28+sh, 0-sh], [28+sh, 28+sh]]
        src, dst = _sort_src_from_dst(rect, dst)

        M = cv2.getPerspectiveTransform(src, dst)
        box = cv2.warpPerspective(box, M, tuple(BOX_SIZE))
        box = np.asarray(box>=127, dtype='uint8')*255
        

        boxes[i] = box
    return boxes
