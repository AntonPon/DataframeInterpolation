import numpy as np

from keras.models import Sequential
from keras.layers import Conv2DTranspose, Conv2D
from keras.optimizers import RMSprop


def build_model():
    model = Sequential()
    model.add(Conv2D(1,  (1, 1), activation='relu', input_shape=(1024, 2, 1)))
    model.add(Conv2DTranspose(512, (1, 1), activation='relu'))
    model.add(Conv2DTranspose(256, (5, 5), activation='relu'))
    model.add(Conv2DTranspose(128, (5, 5), activation='relu'))
    model.add(Conv2DTranspose(64, (5, 5), activation='relu'))
    model.add(Conv2DTranspose(3, (5, 5), activation='relu'))


    return model




# 482 x 360

def patches_correlation(left_image, right_image):
    return left_image * right_image
