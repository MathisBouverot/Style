import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend

# Preprocess an image before it is feeded through VGG19
def img2imgnet(img):
        imgnet = np.copy(img)
        imgnet = imgnet.astype(np.float32)

        # Substract the mean of the ImageNet dataset
        imgnet[ : , : , 0] -= 103.939
        imgnet[ : , : , 1] -= 116.779
        imgnet[ : , : , 2] -= 123.68

        # Reverse color channels from RGB to BGR
        #imgnet = imgnet[ : , : , ::-1]

        return imgnet

def imgnet2img(imgnet):
        img = np.copy(imgnet)

        # Reverse color channels from BGR to RGB
        #img = img[ : , : , ::-1]

        # Add the mean of the ImageNet dataset
        img[ : , : , 0] += 103.939
        img[ : , : , 1] += 116.779
        img[ : , : , 2] += 123.68

        # Clip
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

# Euclidian norm and frobenius norm are the same,
# but euclidian norm is intended for vectors
# and frobenius norm for matrices
def mse(x, y):
        return tf.reduce_mean(tf.square(x - y))

# Compute the gram matrix of a set of features
def gram_matrix(x, y):
        """
        shift = -1
        features = tf.reshape(x, tf.shape(x)[ 1 : ])
        features = backend.batch_flatten(backend.permute_dimensions(features, (2, 0, 1)))
        return backend.dot(features + shift, backend.transpose(features + shift))
        """


        y_up = tf.image.resize_images(y, tf.shape(x)[ 1 : 3])

        shift = -1

        features_A = tf.reshape(x, tf.shape(x)[ 1 : ]) + shift
        features_A = backend.batch_flatten(backend.permute_dimensions(features_A, (2, 0, 1)))

        features_B = tf.reshape(y_up, tf.shape(y_up)[ 1 : ]) + shift
        features_B = backend.batch_flatten(backend.permute_dimensions(features_B, (2, 0, 1)))

        gram = backend.dot(features_A, backend.transpose(features_B))
        return gram
