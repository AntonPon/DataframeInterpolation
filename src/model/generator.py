import numpy as np

from keras.models import Sequential
from keras.layers import Conv2DTranspose, Conv2D, Conv1D, Deconvolution2D
from keras.optimizers import RMSprop


def build_model():
    model = Sequential()
    #model.add(Conv1D(12,   1, activation='relu', input_shape=(1024, 1)))
    model.add(Conv2DTranspose(512, (1, 1), activation='relu', input_shape=(1, 1024, 1)))
    model.add(Conv2DTranspose(256, (7, 7), activation='relu'))
    model.add(Conv2DTranspose(128, (5, 5), activation='relu'))
    model.add(Conv2DTranspose(64, (5, 5), activation='relu'))
    model.add(Deconvolution2D())
    #ranspose(3, (3, 3), activation='relu', output_shape=(360, 482, 3)


    return model




# 482 x 360

def patches_correlation(left_image, right_image):
    return left_image * right_image
