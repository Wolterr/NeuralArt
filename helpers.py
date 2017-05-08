import numpy as np
from scipy import misc

"""
Helper functions for our neural artistic style program. Mainly contains image loading and saving.

"""
# mean pixel values used in VGG. hardcoded because can't be extracted from model.
MEAN_VALUES = np.array([123.68, 116.779, 103.939])


###########################
#  Image functions
###########################
def load_image(image, size=None, shape=None):
    # load image and fix mean values to work with imagenet
    im = misc.imread(image).astype(np.float)

    # resize image if needed
    if size is not None:
        h, w, c = im.shape
        if h > w:
            shape = (size, int(size*(w/h)), c)
        else:
            shape = (int(size*(h/w)), size, c)

    im = misc.imresize(im, shape, interp='bilinear')
    im = im - MEAN_VALUES

    # return resized image + additional dimension that was lost
    return np.reshape(im, (1, shape[0], shape[1], shape[2])), shape


def save_image(tensor, filename, shape):
    # add MEAN_VALUES to image to return to normal RGB space, then save image to disc
    image = tensor.reshape(shape[1:])
    image = image + MEAN_VALUES

    image = np.clip(image, 0, 255).astype('uint8')
    misc.imsave(filename, image)


###########################
#  MISC functions
###########################
def set_input(image=None, shape=None):
    if image is not None:
        return image
    else:
        return np.random.uniform(0, 255, shape) - MEAN_VALUES
