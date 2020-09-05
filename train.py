import glob

import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from sklearn.cross_validation import train_test_split

from data_aug_pipeline import *
from config import *


AUTO = tf.data.experimental.AUTOTUNE
DROPOUT = 0.1



LR_START = 0.0005
LR_MAX = 0.002
LR_MIN = 0.000001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 15
LR_EXP_DECAY = 0.9

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30)


class MnistModel:
    def __init__(self, IMAGE_SIZE, batch_size, epochs, plot=True):
        self.IMAGE_SIZE = IMAGE_SIZE
        self.batch_size = batch_size
        self.epochs = epochs
        self.plot = plot

    def load_data(self):
        # (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        img_files = glob.glob('{}/*/*.jpg'.format(TRAIN_DATA_DIR))
        ds_size = len(img_files)
        x = np.zeros((ds_size, *BOX_SIZE), dtype='uint8')
        y = np.zeros((ds_size, ), dtype='uint8')
        for i, file in enumerate(img_files):
            x[i] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            y[i] = int(file.split('/')[1])
        
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=0.2)
        
    

    def create_train_dataset(self):
        self.train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_ds = self._create_dataset(self.train_ds)
        return self.train_ds
    
    def create_val_dataset(self):
        self.val_ds = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.val_ds = self._create_dataset(self.val_ds)
        return self.val_ds
    

    def get_model(self):
        if len(glob.glob(SAVED_MODEL+"/*")) != 0:
            return tf.keras.models.load_model(SAVED_MODEL)
        self.load_data()
        self.create_train_dataset()
        self.create_val_dataset()
        model = self._build_model()
        hist = self._train_model(model)
        if self.plot:
            self._plot(hist)
        model.save(SAVED_MODEL)
        return model 


    def _plot(self, hist):
        pass

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(self.IMAGE_SIZE),
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1), output_shape=[*IMAGE_SIZE, 1]),
            tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, 'relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(512, 'relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(10, 'softmax')
        ])
        model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    
    def _train_model(self, model):
        return model.fit(self.train_ds, 
           steps_per_epoch=self.x_train.shape[0]//self.batch_size, 
           epochs=self.epochs,
          callbacks = [lr_callback, es_callback],
          validation_data=self.val_ds,
          validation_steps=self.x_val.shape[0]//self.batch_size,
            verbose=1)

            
    def _create_dataset(self, ds):
        with tf.device('/device:GPU:0'):
            ds = ds.map(preprocess, AUTO)
            # ds = ds.map(zeros_to_blank, AUTO)
            ds = ds.repeat()
            # ds = ds.map(random_morph, AUTO)
            ds = ds.map(data_aug, AUTO)
            # ds = ds.map(random_edge_lines, AUTO)
            ds = ds.shuffle(self.batch_size*8)
            ds = ds.batch(self.batch_size, True)
        return ds
