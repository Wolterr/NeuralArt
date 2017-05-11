from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

from model import *
from helpers import load_image, save_image, set_input

"""
A Tensorflow implementation of the paper A Neural Algorithm for Artistic Style
 https://arxiv.org/abs/1508.06576

Program/file structure as following:

- Neural-art.py --> main flow. parsing arguments, building model
- helpers.py --> helper functions. mainly image read and save
- model.py --> model definition and building

"""

#############################
# defaults configuration
#############################
STYLE_INPUT = './images/style/Picasso_Women-of-Algiers.jpg'
CONTENT_INPUT = './images/content/content.jpg'
OUTPUT_LOCATION = './output/output'

WEIGHTS_FILE = './resources/vgg19.npy'

CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e0

LEARNING_RATE = 2
NUM_ITERATIONS = 5000
IMAGE_SIZE = 512

output = ''
PRINT_ITER = 50
SAVE_ITER = 100

CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')
STYLE_LAYER_WEIGHTS = [1, 1, 1, 1, 1]

INPUT_TYPE = 'image'

INIT_OP = tf.global_variables_initializer()


####################################
# Functions for activations and loss
# TODO: Put this into the helpers file.
####################################
def normalize(value_list):
    sum_of_values = sum(value_list)
    normalised = []
    for i in value_list:
        normalised.append(i/sum_of_values)
    return np.array(normalised)

# TODO: Explain gramm matrices and why they are usefull for this
def gram_matrix(activations):
    if type(activations) is np.ndarray:
        activation_dimensions = np.shape(activations)
    else:
        activation_dimensions = activations.get_shape().as_list()
    area = activation_dimensions[1]*activation_dimensions[2]
    depth = activation_dimensions[3]
    gram = tf.reshape(activations, (area, depth))
    return tf.matmul(gram, gram, transpose_a=True)

#############################
# Parse inputs
#############################
parser = argparse.ArgumentParser(description='Neural algorithm style transfer with Keras and Tensorflow.')

# basic options
parser.add_argument('--style-image', dest='style', help='style image', default=STYLE_INPUT, type=str)
parser.add_argument('--content-image', dest='content', help='content image', default=CONTENT_INPUT, type=str)
parser.add_argument('--image_size', dest='size', help='maximum width and/or height of the image', default=IMAGE_SIZE, type=int)
parser.add_argument('--input-type', dest='input_type', help='use either content image or random noise as input', default=INPUT_TYPE, type=str)


# output options
parser.add_argument('--output', dest='output', help='output name', default=OUTPUT_LOCATION, type=str)
parser.add_argument('--print-iter', dest='print_iter', help='interval to print training information', default=PRINT_ITER, type=int)
parser.add_argument('--save-iter', dest='save_iter', help='interval to save generated images', default=SAVE_ITER, type=int)

# optimization options
parser.add_argument('--learning-rate', dest='learning_rate', help='learning rate', default=LEARNING_RATE, type=float)
parser.add_argument('--content-weight', dest='content_weight', help='content weight factor', default=CONTENT_WEIGHT, type=float)
parser.add_argument('--style-weight', dest='style_weight', help='style weight factor', default=STYLE_WEIGHT, type=float)
parser.add_argument('--variation-weight', dest='tv_weight', help='axis-wise variation normalisation weight factor', default=TV_WEIGHT, type=float)
parser.add_argument('--iterations', dest='iterations', help='number of iterations', default=NUM_ITERATIONS, type=int)

# layer options
parser.add_argument('--content-layer', dest='content_layers', help='layer for content representation', default=CONTENT_LAYER, type=str)
parser.add_argument('--style-layers', dest='style_layers', help='style representation layers to use', default=STYLE_LAYERS, type=str)
parser.add_argument('--style-layer-weights', dest='style_layer_weights', help='weights for each style layer to use', default=STYLE_LAYER_WEIGHTS,
                    type=list)
# TODO: Fix style layer and style layer weights types so these cmd-line options actually work.

parser.add_argument('--weights-file', dest='weights_file', help='File to read pre-trained VGG19 weights from', default=WEIGHTS_FILE, type=str)

# Set options from arguments
options = parser.parse_args()

OUTPUT_LOCATION = options.output
LEARNING_RATE = options.learning_rate
NUM_ITERATIONS = options.iterations
IMAGE_SIZE = options.size
INPUT_TYPE = options.input_type

# Normalize the content, style and variation weight.
CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT = normalize([options.content_weight, options.style_weight, options.tv_weight])

CONTENT_LAYER = options.content_layers
STYLE_LAYERS = options.style_layers
STYLE_LAYER_WEIGHTS = dict(zip(STYLE_LAYERS, options.style_layer_weights)) # create a dict so we do not have to rely on list ordering to be the
# same later on

PRINT_ITER = options.print_iter
SAVE_ITER = options.save_iter

WEIGHTS_FILE = options.weights_file

##################################
#           Main flow
##################################
# set input images
print("{} : Set input images".format(time.strftime("%c")))
content_image, image_shape = load_image(options.content, size=IMAGE_SIZE)
style_image, _ = load_image(options.style, size=IMAGE_SIZE)

# Change image_shape to be usable for our network layers
image_shape = (1, image_shape[0], image_shape[1], image_shape[2])

########################################
# Pre-compute content and style targets
########################################

# Compute initial content activation
#
# The content activation is saved in an ndarray of size of CONTENT_LAYER. These are RELU activations for that layer.
#
# Works as following: Build a model using the model definition in the model.py file. The content image is used as the input.
# The build network is then evaluated at the CONTENT_LAYER, doing a feed-forward run on the network.
#
print("{} : Compute initial content activation".format(time.strftime("%c")))
content_activations = None
with tf.Graph().as_default(), tf.Session():
    image = tf.placeholder('float', shape=image_shape)
    model = build_model(WEIGHTS_FILE, image)

    content_activations = model[CONTENT_LAYER].eval(feed_dict={image: content_image})

# Compute initial style activation
#
# The style activations are saved in a dict. These are the per-layer Gramm matrices.
#
# This works are following: Build a model using the style_image as input layer. For each of the specified STYLE_LAYERS do a feed-forward pass from
# the style image to the specific layer. Using the activations, calculate the gramm matrix and append this to our dict.
#
# Note: We probably do not need a new graph here, but this ensures there are no residuals from our content activation calculation.
#
print("{} : Compute initial style activations".format(time.strftime("%c")))
style_activations = {}
with tf.Session():
    image = tf.placeholder('float', shape=style_image.shape)
    model = build_model(WEIGHTS_FILE, image)

    for layer in STYLE_LAYERS:
        layer_activations = model[layer].eval(feed_dict={image: style_image})
        style_activations[layer] = gram_matrix(layer_activations).eval()

###########################################################################
# Define the computation graph for training our model to output nice images
###########################################################################

with tf.Graph().as_default(), tf.Session() as sess:
    print("{} : Set the input tensor and create image variable".format(time.strftime("%c")))
    input_tensor = set_input(content_image) if INPUT_TYPE == 'image' else set_input(shape=image_shape)
    image = tf.Variable(input_tensor, dtype='float')

    print("{} : Build model we want to train".format(time.strftime("%c")))
    model = build_model(WEIGHTS_FILE, image)

    sess.run(tf.global_variables_initializer())

    # Calculate content loss
    #
    # Calculate difference between the originally computed activations and the current activation in the CONTENT_LAYER - minimizing layer
    # activation differences.
    #
    # Using a normalization factor of the total amount of activations - dividing sum by image dimensions * number of filters - instead of the
    # factor 2 (standard l2_loss) as used in the paper.
    #
    # Using l2_loss meant having to use style weights in the order of 1e9 vs 5e0 for content.
    # Several other implementations i've seen used 4*h*w*c, but there doesn't seem to be a good explanation for the factor 4 so this is left
    # out.
    #
    print("{} : Calculate content loss".format(time.strftime("%c")))
    diff = model[CONTENT_LAYER] - content_activations
    _, h, w, n = content_activations.shape
    loss_content = tf.nn.l2_loss(diff, name='content_loss') * 2 / (h*w*n)

    # Calculate style loss
    #
    # Using formula as used in the paper: normalizing each layer by quite a large factor. Left factor 4 in here because the paper uses it as well.
    #
    # Minimizing the style loss means minimizing the difference between the gramm (~auto-correlations per filter) matrices.
    #
    print("{} : Calculate style loss".format(time.strftime("%c")))
    style_losses = []
    for layer in STYLE_LAYERS:
        layer_activations = model[layer]
        shape = layer_activations.get_shape().as_list()

        k = 4 * shape[3]**2 * (shape[1]*shape[2])**2

        diff = gram_matrix(layer_activations) - style_activations[layer]

        layer_losses = tf.reduce_sum(diff**2) / k
        style_losses.append(layer_losses * STYLE_LAYER_WEIGHTS[layer])

    loss_style = tf.reduce_sum(style_losses)

    # total variation loss
    #
    # Although it isn't mentioned in the paper the, I'm assuming, original Lua implementation (https://github.com/jcjohnson/neural-style) of the
    # algorithm uses a variational denoising term in the loss function. Using this does indeed get us closer to the images created by their
    # implementation and the site deepart.io.
    #
    # The idea behind total variation denoising:
    # TODO: Read up on this and explain here.
    #
    loss_tv = tf.image.total_variation(tf.reshape(image, [image_shape[1], image_shape[2], image_shape[3]]))

    # Calculate overall loss
    #
    # Using standard formula from paper. Added simple variation denoising terms.
    #
    loss_overall = (CONTENT_WEIGHT * loss_content) + (STYLE_WEIGHT * tf.cast(loss_style, dtype=tf.float32)) + (TV_WEIGHT * loss_tv)

    # Set the optimizer and train operations
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss_overall)

    # Initialise everything
    sess.run(tf.global_variables_initializer())

    #############################
    # Run training and synthesis
    #############################
    # The training iterates until NUM_ITERATIONS.
    #
    # The following output is generated
    #   - Loss information every 50 iterations
    #   - A stylized image every 100 iterations. Saved to the output location, adding iteration number and ".jpg" at the end.
    #
    # TODO: Add flag to only save best image
    #
    for iteration in range(NUM_ITERATIONS+1):
        train_op.run()
        if iteration % PRINT_ITER == 0:
            print("At iteration {} \ Loss = {:.5e} \ Content loss = {:.5e} \ Style loss = {:.5e} \ Variation loss = {:.5e}".format(iteration,
                                                                                                                        loss_overall.eval(),
                                                                                                         loss_content.eval(), loss_style.eval(),
                                                                                                         loss_tv.eval()))
        if iteration % SAVE_ITER == 0:
            im = image.eval()
            save_image(im, OUTPUT_LOCATION + str(iteration) + ".jpg", shape=image_shape)

print("########################################################")
print("# Done processing image. Go look in the output folder! #")
print("########################################################")
