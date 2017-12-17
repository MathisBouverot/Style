
# Imports
import numpy as np
import tensorflow as tf

import matplotlib
from matplotlib import pyplot as plt

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import MaxPooling2D, AveragePooling2D, Input, Conv2D
from tensorflow.python.keras.models import Model

import cv2
import utils
from scipy.optimize import minimize
import time

# Session
sess = tf.Session()
K.set_session(sess)

# Hyperparameters
WIDTH = 350
HEIGHT = 350

CONTENT_PATH = 'img/content.jpg'
STYLE_PATH = 'img/bulles.jpg'

CONTENT_WEIGHT = tf.Variable(0.0)
STYLE_WEIGHT = tf.Variable(0.0)
TOTAL_VARIATION_WEIGHT = tf.Variable(0.0)


# Read the images
content_img = cv2.imread(CONTENT_PATH, cv2.IMREAD_COLOR)
content_img = cv2.resize(content_img, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)

style_img = cv2.imread(STYLE_PATH, cv2.IMREAD_COLOR)
style_img = style_img[  : 600, : 600 , : ]
style_img = cv2.resize(style_img, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
cv2.imshow('style', style_img)
cv2.waitKey(0)

content_img = utils.img2imgnet(content_img)
style_img = utils.img2imgnet(style_img)

# Create the input tensor to the network
#creation = tf.placeholder(tf.float32, shape = (HEIGHT, WIDTH, 3))
#content = tf.Variable(content_img)
#style = tf.Variable(style_img)

creation = tf.placeholder(tf.float32, shape = (1, HEIGHT, WIDTH, 3))

# Create the model
with tf.variable_scope('vgg'):
        with tf.variable_scope('model'):
                model_original = VGG16(input_tensor = creation, include_top = False, weights = 'imagenet')
                #model_original.summary()

                # Replace the max pooling layers with average pooling
                inp = model_original.input

                out = inp
                for i in range(1, len(model_original.layers)): # don't include the input layer
                        lay = model_original.layers[i]
                        config = lay.get_config()
                        #print(config)
                        if type(lay) == MaxPooling2D:
                                #print("Pooling")
                                #out = AveragePooling2D(
                                #        pool_size = lay.pool_size,
                                #        strides = lay.strides,
                                #        padding = lay.padding)(out)
                                new_lay = AveragePooling2D.from_config(config)
                                out = new_lay(out)
                        else:
                                new_lay = Conv2D.from_config(config)
                                #print(new_lay.get_config())
                                #print(new_lay.get_weights())
                                #print(new_lay.filters, new_lay.kernel_size)
                                out = new_lay(out)
                                new_lay.set_weights(lay.get_weights())

                model = Model(inp, out)
                model.summary()

# Define the loss

# Content loss
content_layers = ['block4_conv2']
content_loss_op = tf.Variable(0.0)

coef = 1
for i in range(len(content_layers)):
        layer_name = content_layers[i]

        content_features = model.get_layer(layer_name).output
        content_features = sess.run(content_features, feed_dict = {creation: np.expand_dims(content_img, 0)})
        content_features = tf.constant(content_features)

        creation_features = model.get_layer(layer_name).output
        weight = coef ** i
        content_loss_op += weight * utils.mse(content_features, creation_features)

content_loss_op /= float(len(content_layers))

# Style loss
style_layers = [
        'block1_conv1', #'block1_conv2',
        'block2_conv1', #'block2_conv2',
        'block3_conv1', #'block3_conv2', 'block3_conv3',
        'block4_conv1', #'block4_conv2', 'block4_conv3',
        'block5_conv1'
]
style_loss_op = tf.Variable(0.)

coef = 1
for i in range(0, len(style_layers)):
        layer_name_A = style_layers[i]
        style_features_A = model.get_layer(layer_name_A).output
        style_features_A = sess.run(style_features_A, feed_dict = {creation : np.expand_dims(style_img, 0)})
        style_features_A = tf.constant(style_features_A)

        layer_name_B = style_layers[i - 1]
        style_features_B = model.get_layer(layer_name_B).output
        style_features_B = sess.run(style_features_B, feed_dict = {creation : np.expand_dims(style_img, 0)})
        style_features_B = tf.constant(style_features_B)

        creation_features_A = model.get_layer(layer_name_A).output

        creation_features_B = model.get_layer(layer_name_B).output

        S = utils.gram_matrix(style_features_A, style_features_B)
        C = utils.gram_matrix(creation_features_A, creation_features_B)
        weight = coef ** (len(style_layers) - i)
        style_loss_op += weight * utils.mse(S, C) / float((WIDTH * HEIGHT) ** 2)

style_loss_op /= float(len(style_layers))

# Total variation loss
a = tf.square(creation[ : , : HEIGHT - 1, : WIDTH - 1, : ] - creation[ : , 1 : , : WIDTH - 1, : ])
b = tf.square(creation[ : , : HEIGHT - 1, : WIDTH - 1, : ] - creation[ : , : HEIGHT - 1, 1 : , : ])
total_variation_loss_op =  tf.reduce_mean(tf.pow(a + b, 1.25))

total_variation_loss_op *= TOTAL_VARIATION_WEIGHT

style_loss_op *= STYLE_WEIGHT
content_loss_op *= CONTENT_WEIGHT

loss_op = content_loss_op + style_loss_op + total_variation_loss_op


grad_op = tf.gradients(xs = [creation], ys = loss_op)

# Compute the loss and gradient of the loss
# with respect to the input image x (creation)
def compute_loss_and_grads(x):
        x = x.reshape((1, HEIGHT, WIDTH, 3))
        fetches = [loss_op, grad_op]
        feed_dict = {creation : x}
        loss, grads = sess.run(fetches, feed_dict = feed_dict)
        grad = grads[0] # we actually computed the gradients w.r.t. [x], not w.r.t. x
        return loss, grad.astype(np.float64).flatten()

new_cw = tf.placeholder(tf.float32)
update_cw = tf.assign(CONTENT_WEIGHT, new_cw)

new_sw = tf.placeholder(tf.float32)
update_sw = tf.assign(STYLE_WEIGHT, new_sw)

new_tvw = tf.placeholder(tf.float32)
update_tvw = tf.assign(TOTAL_VARIATION_WEIGHT, new_tvw)

#x = np.random.uniform(0, 255, (HEIGHT, WIDTH, 3)) - 128.
#x = x.flatten()
x = np.copy(content_img) + np.random.uniform(-30, 30, (HEIGHT, WIDTH, 3))
x = x.flatten()

#init_op = tf.global_variables_initializer()
#sess.run(init_op)

ITERATIONS = 100
"""
w = model.get_layer('block1_conv1').get_weights()[0]
plt.hist(w.flatten())
plt.title("w values after 'imagenet' weights have been loaded")
plt.show()
"""
vgg_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'vgg/model')

saver = tf.train.Saver(vgg_weights)
checkpoint_path = saver.save(sess, '/tmp/vgg_weights.ckpt')

sess.run(tf.global_variables_initializer())

saver.restore(sess, checkpoint_path)
"""
w = model.get_layer('block1_conv1').get_weights()[0]
plt.hist(w.flatten())
plt.title("w values after tf.global_variables_initializer()")
plt.show()
"""

for it in range(ITERATIONS):
        print('*******Iteration %d*******' % it)
        img = x.reshape((HEIGHT, WIDTH, 3))
        img = utils.imgnet2img(img)

        cv2.imshow('creation', img)
        cv2.waitKey(0)

        cw = float(input("content weight : "))
        sw = float(input("style weight : "))
        tvw = float(input("total variation weight : "))
        maxfun = int(input("maxfun : "))

        sess.run(update_cw, feed_dict = {new_cw : cw})
        sess.run(update_sw, feed_dict = {new_sw : sw})
        sess.run(update_tvw, feed_dict = {new_tvw : tvw})
        start = time.time()

        # minimize expects the image and the gradient to have rank one
        res = minimize(compute_loss_and_grads, x, method = 'L-BFGS-B', jac = True, options = {'maxfun' : maxfun})
        x = res.x
        loss = res.fun

        end = time.time()

        print('\tElapsed time : %.2fs' % (end - start))
        print('\tLoss : %.4f' % loss)


        fetches = [style_loss_op, content_loss_op, total_variation_loss_op]
        feed_dict = {creation : x.reshape((1, HEIGHT, WIDTH, 3))}
        sl, cl, tvl = sess.run(fetches, feed_dict = feed_dict)
        print("\tStyle loss : %.2f\n\tContent loss : %.2f\n\tTotal variation loss : %.2f" % (sl, cl, tvl))



cv2.destroyAllWindows()
sess.close()
