# Imports
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
import cv2

INCEPTION_PATH = 'inception/tensorflow_inception_graph.pb'

graph = tf.Graph()
sess = tf.InteractiveSession(graph = graph)

with tf.gfile.FastGFile(INCEPTION_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


inp = tf.placeholder(np.float32, name = 'input')
imagenet_mean = 117.0

preprocessed = tf.expand_dims(inp - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input' : preprocessed})

layers = {}
for op in graph.get_operations():
        if op.type == 'Conv2D':
                #print(op.name, op.outputs[0])
                layers[op.name] = op.outputs[0]

print('\n'.join(list(layers.keys())[-5 : ]))

def show_img(img):
        img = np.clip(img, 0, 255).astype(np.uint8)
        plt.imshow(img)
        plt.show()


def deep_dream_naive(layer_name, channel, iterations, lr, start_img):
        tv_weight = 0.001
        reconstruct_weight = 1.0

        features = layers[layer_name]
        #features = graph.get_tensor_by_name("import/%s:0" % layer_name)
        features = features[ : , : , : , channel]

        reconstruct_score_op = tf.reduce_mean(features)

        a = tf.square(inp[ : -1, : -1, : ] - inp[ 1 : , : -1, : ])
        b = tf.square(inp[ : -1, : -1, : ] - inp[ : -1, 1 : , : ])
        tv_loss_op = tf.reduce_mean(tf.pow(a + b, 1.25))

        score_op = reconstruct_weight * reconstruct_score_op - tv_weight * tv_loss_op
        grad_op = tf.gradients(xs = inp, ys = score_op)[0]

        print('Deep Dream Start')
        creation = np.copy(start_img)
        start = time.time()
        for it in range(iterations):
                grad, score, tv_loss = sess.run([grad_op, score_op, tv_loss_op], feed_dict = {inp : creation})
                print(tv_loss)
                grad /= grad.std() + 1e-8
                #print(grad)
                creation += grad * lr

                if it % 30 == 0 and it != 0:
                        end = time.time()
                        print("Iteration %d : elapsed time %.2fs and score %.5f" % (it, end - start, score))
                        show_img(creation)
                        start = time.time()

        print('\nFinished')
        return creation

def deep_dream_octaves(layer_name, channel, iterations, lr, start_img, octaves = 4, octave_scale = 1.4):
        reconstruct_weight = 1.0
        tv_weight = 0.002

        features = layers[layer_name][ : , : , : , channel]

        reconstruct_score_op = tf.reduce_mean(features)

        a = tf.square(inp[ : -1, : -1, : ] - inp[ 1 : , : -1, : ])
        b = tf.square(inp[ : -1, : -1, : ] - inp[ : -1, 1 : , : ])
        tv_loss_op = tf.reduce_mean(tf.pow(a + b, 1.25))

        score_op = reconstruct_weight * reconstruct_score_op - tv_weight * tv_loss_op
        grad_op = tf.gradients(xs = inp, ys = score_op)[0]

        creation = np.copy(start_img)
        start = time.time()
        for oc in range(octaves):
                if oc != 0:
                        new_dim = (
                                np.uint16(np.float32(creation.shape[1]) * octave_scale),
                                np.uint16(np.float32(creation.shape[0]) * octave_scale)
                        )
                        print('New dim ', new_dim)
                        creation = cv2.resize(creation.astype(np.uint16), new_dim, interpolation = cv2.INTER_AREA)
                        creation = creation.astype(np.float32)
                for it in range(iterations):
                        #grad = calculate_grad_tiled(creation, grad_op)
                        grad = sess.run(grad_op, feed_dict = {inp : creation})
                        grad /= grad.std() + 1e-8
                        creation += grad * lr
                        creation = np.clip(creation, 0, 255)

                        if it % 10 == 0 and it != 0:
                                end = time.time()
                                score = sess.run(score_op, feed_dict = {inp : creation})
                                print("Iteration %d : elapsed time %.2fs and score %.5f" % (it, end - start, score))
                                if oc >= 2:
                                        show_img(creation)
                                start = time.time()

        return creation

layer_name = 'import/mixed4d_3x3_bottleneck_pre_relu/conv'

"""
start_img = cv2.imread('img/content.jpg', cv2.IMREAD_COLOR)
start_img = cv2.resize(start_img, (200, 200), interpolation = cv2.INTER_AREA)
start_img = np.flip(start_img, 2) # convert from BGR to RGB
start_img = start_img.astype(np.float32)
"""

start_img = np.random.uniform(0, 255, size = (200, 200, 3)).astype(np.float32)
#show_img(start_img)

creation = deep_dream_octaves(
        layer_name, channel = 139, iterations = 31, lr = 3.0, start_img = start_img,
        octaves = 7, octave_scale = 1.4
)

show_img(creation)

sess.close()


# flowers : import/mixed4d_3x3_bottleneck_pre_relu/conv    139
# chihuahuas : import/mixed4d_3x3_bottleneck_pre_relu/conv    1
# arches : import/mixed4d_3x3_bottleneck_pre_relu/conv    2
