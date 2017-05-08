# A Neural Algorithm for Artistic Style

This repository contains a tensorflow implementation of the paper [A Neural Algorithm for Artistic Style](http://arxiv.org/abs/1508.06576) by Leon. A
Gatys, et all.

The idea behind the paper and this implementation is that using the, theoretically strong and stable, internal represantations of convolutional
neural networks, we can generate "new" patterns.
This work uses a pre-trained network (in this case VGG-19)


# Examples
Lets start with the mandatory Tubingen (germany) + starry sky combination.
<div align="center">
 <img src="https://raw.githubusercontent.com/wolterr/NeuralArt/master/examples/content/tubingen.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/wolterr/NeuralArt/master/examples/style/starry_night.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/Wolterr/NeuralArt/master/examples/output/starry_tubingen.jpg" width="512px">
</div>

Next a Cubism style image of a street in Trinidad (Cuba)
<div align="center">
 <img src="https://raw.githubusercontent.com/wolterr/NeuralArt/master/examples/content/trinidad.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/wolterr/NeuralArt/master/examples/style/cubism.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/wolterr/NeuralArt/master/examples/output/cubed_trinidad.jpg" width="512px">
</div>

# Usage
Basic usage:

'''
python neuralArt.py --content-image <CONTENT> --style-image <STYLE> --output <OUTPUTLOCATION>
'''


## optional arguments:
  '-h, --help'                                    show this help message and exit
  '--style-image STYLE'                           style image
  '--content-image CONTENT'                       content image
  '--image_size SIZE'                             maximum width and/or height of the image
  '--input-type INPUT_TYPE'                       use either content image or random noise as input
  '--output OUTPUT'                               output name
  '--print-iter PRINT_ITER'                       interval to print training information
  '--save-iter SAVE_ITER'                         interval to save generated images
  '--learning-rate LEARNING_RATE'                 learning rate
  '--content-weight CONTENT_WEIGHT'               content weight factor
  '--style-weight STYLE_WEIGHT'                   style weight factor
  '--variation-weight TV_WEIGHT'                  axis-wise variation normalisation weight factor
  '--iterations ITERATIONS'                       number of iterations
  '--content-layer CONTENT_LAYERS'                layer for content representation
  '--style-layers STYLE_LAYERS'                   style representation layers to use
  '--style-layer-weights STYLE_LAYER_WEIGHTS'     weights for each style layer to use
  '--weights-file WEIGHTS_FILE'                   File to read pre-trained VGG19 weights from

# Acknowledgements
* The idea and implementation are based on the [paper](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker & Matthias Bethge.
* The pre-trained weights used in the model are from a [tensorflow port](https://github.com/machrisaa/tensorflow-vgg) of the Caffe VGG-19 weights,
done by [machrisaa](https://github.com/machrisaa).
* Idea's to match the output at deepart.io more closely after initial implementation were found in the original [Lua implementation](https://github
.com/jcjohnson/neural-style).
* Some of the code is based on an implementation by [anishathalye](https://github.com/anishathalye/neural-style).