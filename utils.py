""" Helper function for the main pipeline """

# Imports
import numpy as np
import torch
import copy
import matplotlib

def generate_datasets(data_stack, train_val_fraction=0.8):
    """
    Generate train and validation data stacks

    Parameters
    ----------
    data_stack : numpy.ndarray
        image data stack used for the datasets
    train_val_fraction : float, optional
        size fraction between train and validation datasets
        (default: 0.8)

    Returns
    -------
    train_data : numpy.ndarray
        train data stack
    validation_data : numpy.ndarray
        validation data stack
    """

    shuffled_index = np.arange(len(data_stack))
    np.random.shuffle(shuffled_index)

    train_data = data_stack[shuffled_index[:int(train_val_fraction*len(data_stack))]]
    validation_data = data_stack[shuffled_index[int(train_val_fraction*len(data_stack)):]]

    return [train_data], [validation_data]

def generate_3Ddatasets(data_stack, train_val_fraction=0.8):
    """
    Generate train and validation data stacks

    Parameters
    ----------
    data_stack : numpy.ndarray
        image data stack used for the datasets
    train_val_fraction : float, optional
        size fraction between train and validation datasets
        (default: 0.8)

    Returns
    -------
    train_data : numpy.ndarray
        train data stack
    validation_data : numpy.ndarray
        validation data stack
    """

    train_data = data_stack[:int(train_val_fraction*len(data_stack))]
    validation_data = data_stack[int(train_val_fraction*len(data_stack)):]

    return [train_data], [validation_data]

# Tensorboard just clip the input data between 0 and 1, but don't adjust the
# range of the images
# For this reason the images get encoded in the matplotlib gray-colormap and
# are passed to tensorboard afterwards
class ChangeColormap4Tensorboard(object):

    def __init__(self, neglect_alpha=True, cmap=None):
        self.neglect_alpha = neglect_alpha
        self.cmap = cmap

    # see: https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
    def colorize(self, value):
        """
        A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
        colormap for use with TensorBoard image summaries.
        By default it will normalize the input value to the range 0..1 before mapping
        to a grayscale colormap.
        Arguments:
          - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
            [height, width, 1].
          - vmin: the minimum value of the range used for normalization.
            (Default: value minimum)
          - vmax: the maximum value of the range used for normalization.
            (Default: value maximum)
          - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
            (Default: Matplotlib default colormap)

        Returns a 4D uint8 tensor of shape [height, width, 4].
        """

        neglect_alpha = self.neglect_alpha
        vmin = self.vmin
        vmax = self.vmax
        cmap = self.cmap

        # normalize
        vmin = value.min() if vmin is None else vmin
        vmax = value.max() if vmax is None else vmax
        if vmin != vmax:
            value = (value - vmin) / (vmax - vmin) # vmin..vmax
        else:
            # Avoid 0-division
            value = value*0.
        # squeeze last dim if it exists
        value = value.squeeze()

        cmapper = matplotlib.cm.get_cmap(cmap)
        value = cmapper(value, bytes=True) # (nxmx4)

        if neglect_alpha:
            value = value[:, :, 0:3]

        return value

    def __call__(self, image, vmin=None, vmax=None):


        self.vmin = vmin
        self.vmax = vmax
        self.image = copy.copy(image)

        return self.colorize(self.image)

# For the later comparison between the data, the network returns, and the
# original images, the output data is changed back to int16-format
class NormFloat2Int16(object):

    def __call__(self, data):
        data = (data - 0.5) * (np.iinfo(np.int16).max + np.abs(np.iinfo(np.int16).min))
        data = np.clip(data, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
        return np.rint(data).astype(np.int16)

# For transformation between float32 ([0, 1]) to uint16 with clipping
class NormFloat2Uint16(object):

    def __call__(self, data):
        data = np.clip(data*np.iinfo(np.uint16).max, 0, np.iinfo(np.uint16).max)
        return np.rint(data).astype(np.uint16)

# Apply a certain method to all data of the batch
class BatchTransformer(object):

    def __call__(self, method, data):
        if isinstance(data, np.ndarray):
            output = np.asarray([method(d) for d in data])#, dtype=object)
        else:
            output = torch.as_tensor([method(d) for d in data])

        return output



