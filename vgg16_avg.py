import tensorflow as tf
import numpy as tf

from tensorflow.python.keras.layers import Convolution2D, AveragePooling2D, Input
from tensorflow.python.keras.models import Model


def vgg16_avg(input_shape):
        img_input = Input(shape = input_shape)

        # Block 1
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
        x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
        x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
        x = AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
