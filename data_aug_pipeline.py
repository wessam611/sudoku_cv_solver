import math

import tensorflow as tf
import tensorflow.keras.backend as K

from config import *


IMG_THRESH = 0.3
RANDOM_SALT_NOISE = 0.05
RANDOM_LINE_NOISE = 0.02


def preprocess(image, label):
    img = tf.cast(image, 'float32') / 255.0
    lbl = tf.one_hot(label, 10)
    img = tf.cast(tf.cast(img + (1. - IMG_THRESH), tf.int32), 'float32') # thresholding
    return img, lbl


# not used in final
def zeros_to_blank(image, label):
    if label[0] == 1:
        image = tf.zeros(tf.shape(image), 'float32')
    return image, label

# not used in final
def random_noise(image, label):
    rand = tf.random.uniform(tf.shape(image), dtype='float32')
    rand = tf.cast(tf.cast(rand+RANDOM_SALT_NOISE, 'int32'), 'bool')
    image = tf.cast(tf.math.logical_xor(tf.cast(image, 'bool'), rand), 'float32')
    return image, label

# not used in final
def random_edge_lines(image, label):
    pad = 2
    ones = tf.ones(pad)
    zeros = tf.zeros((tf.shape(image)[0]-2*pad))
    mask = tf.concat([ones, zeros, ones], 0)
    mask = tf.cast(mask, 'bool')

    rand_h = tf.random.uniform([tf.shape(image)[0]])
    rand_h = tf.cast(tf.cast(rand_h+RANDOM_LINE_NOISE, 'int32'), 'bool')
    rand_h = tf.expand_dims(rand_h, 0)
    rand_h = tf.math.logical_and(rand_h, mask)
    rand_v = tf.random.uniform([tf.shape(image)[1]])
    rand_v = tf.cast(tf.cast(rand_v+RANDOM_LINE_NOISE, 'int32'), 'bool')
    rand_v = tf.expand_dims(rand_v, 0)
    rand_v = tf.math.logical_and(rand_v, mask)

    rand_h = tf.tile(rand_h, [tf.shape(image)[0], 1])
    rand_v = tf.transpose(tf.tile(rand_v, [tf.shape(image)[1], 1]))

    rand = tf.math.logical_or(rand_h, rand_v)

    rand = tf.expand_dims(tf.expand_dims(rand, -1), 0)
    rand = tf.cast(rand,'float32')
    # rand = tf.nn.max_pool2d(rand, ksize=(3, 3), strides=1, padding='SAME')
    rand = tf.nn.dropout(rand, 0.9)
    rand = tf.cast(rand, 'bool')
    rand = tf.squeeze(tf.squeeze(rand, 0), -1)
    

    image = tf.cast(tf.math.logical_or(tf.cast(image, 'bool'), rand), 'float32')

    return image, label


# not used in final
def random_morph(img, label):
    w, h = BOX_SIZE
    im = tf.expand_dims(tf.expand_dims(img, 0), -1)
    morph = tf.random.uniform([2], dtype='float32')*0.7 + 2.3
    morph_d = tf.cast(morph*w, dtype='int32')
    im = tf.image.resize(im, morph_d)
    im = -tf.nn.max_pool2d(-im, ksize=3, strides=1, padding='SAME')
    im = tf.image.resize(im, [w, h])
    im = tf.squeeze(tf.squeeze(im, 0), -1)
    im = tf.cast(tf.cast(im + (1. - IMG_THRESH), tf.int32), 'float32') # thresholding
    return im, label

def _get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )
        
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
    
    # ZOOM MATRIX
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
    
    # SHIFT MATRIX
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
    
    
    return K.dot(K.dot(shift_matrix, shear_matrix), K.dot(rotation_matrix, zoom_matrix))


def data_aug(image, label):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = BOX_SIZE[0]
    XDIM = DIM%2 #fix for size 331
    
    rot = 3. * tf.random.normal([1],dtype='float32')
    shr = 1. * tf.random.normal([1],dtype='float32') 
    h_zoom = 1 + tf.random.normal([1],dtype='float32')*0.07
    # w_zoom = 0.8 + tf.random.normal([1],dtype='float32')*0.3
    w_zoom = h_zoom + tf.random.normal([1],dtype='float32')*0.02
    h_shift = 2. * tf.random.normal([1],dtype='float32') 
    w_shift = 2. * tf.random.normal([1],dtype='float32')
  
    # GET TRANSFORMATION MATRIX
    m = _get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack([x,y,z])
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))
    im = tf.reshape(d,[DIM,DIM])
    return im, label