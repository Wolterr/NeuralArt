import tensorflow as tf
import numpy as np

"""
File for creating the model we need to do the artistic rendering.

The model uses layers based on pre-trained VGG19 weights.
Download the used weights from using pre-trained weights from https://github.com/machrisaa/tensorflow-vgg

Model uses all layers up until conv5_1. This is the highest layers used for image/style reconstruction.
No use in loading the other layers (Conv5_2 until top) when they are not used.

Model uses avg-pooling instead of max-pooling following the paper.
"""


###########################
#  Layer functions
###########################
def create_conv_layer(previous_layer, name, weights):
    return tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(previous_layer, filter=weights[0], strides=[1, 1, 1, 1], padding='SAME'), weights[1]
        ), name=name
    )


def create_pooling_layer(previous_layer, name):
    return tf.nn.avg_pool(previous_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def build_model(weights_file, img):
    model_weights = np.load(weights_file, encoding='latin1').item()

    graph = {}
    #graph['input'] = img

    # turn input into bgr
    # red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=img)
    # bgr = tf.concat(axis=3, values=[blue, green, red])

    # first layer group
    graph['conv1_1'] = create_conv_layer(img, 'conv1_1', model_weights['conv1_1'])
    graph['conv1_2'] = create_conv_layer(graph['conv1_1'], 'conv1_2', model_weights['conv1_2'])
    graph['pool1'] = create_pooling_layer(graph['conv1_2'], 'pool1')


    # second layer group
    graph['conv2_1'] = create_conv_layer(graph['pool1'], 'conv2_1', model_weights['conv2_1'])
    graph['conv2_2'] = create_conv_layer(graph['conv2_1'], 'conv2_2', model_weights['conv2_2'])
    graph['pool2'] = create_pooling_layer(graph['conv2_2'], 'pool2')

    # third layer group
    graph['conv3_1'] = create_conv_layer(graph['pool2'], 'conv3_1', model_weights['conv3_1'])
    graph['conv3_2'] = create_conv_layer(graph['conv3_1'], 'conv3_2', model_weights['conv3_2'])
    graph['conv3_3'] = create_conv_layer(graph['conv3_2'], 'conv3_3', model_weights['conv3_3'])
    graph['conv3_4'] = create_conv_layer(graph['conv3_3'], 'conv3_4', model_weights['conv3_4'])
    graph['pool3'] = create_pooling_layer(graph['conv3_4'], 'pool3')

    # fourth layer group
    graph['conv4_1'] = create_conv_layer(graph['pool3'], 'conv4_1', model_weights['conv4_1'])
    graph['conv4_2'] = create_conv_layer(graph['conv4_1'], 'conv4_2', model_weights['conv4_2'])
    graph['conv4_3'] = create_conv_layer(graph['conv4_2'], 'conv4_3', model_weights['conv4_3'])
    graph['conv4_4'] = create_conv_layer(graph['conv4_3'], 'conv4_4', model_weights['conv4_4'])
    graph['pool4'] = create_pooling_layer(graph['conv4_4'], 'pool4')

    # fifth layer group
    graph['conv5_1'] = create_conv_layer(graph['pool4'], 'conv5_1', model_weights['conv5_1'])
    # We stop here with the model. Higher layers are not used in image construction

    return graph
