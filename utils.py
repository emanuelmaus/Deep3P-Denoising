""" Helper function for the main pipeline """

# Imports
import numpy as np
import torch
import copy
import os
import matplotlib

from scipy.fftpack import fft, fftshift, fftfreq

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

# Apply a certain method to all data of the batch
class BatchTransformer(object):

    def __call__(self, method, data):
        if isinstance(data, np.ndarray):
            output = np.asarray([method(d) for d in data])#, dtype=object)
        else:
            output = torch.as_tensor([method(d) for d in data])

        return output

## Helper function for inference

# Load  trained model
def load(dir_chck, netG, device, epoch=[], use_best=False):

    if not os.path.exists(dir_chck) or not os.listdir(dir_chck):
        print('There is error and no data in dir_check')

    if use_best:
        dict_net = torch.load(os.path.join(dir_chck, 'best_model.pth'), map_location=device)
        print(dict_net.keys())

        epoch = dict_net['epoch']
        print('Loaded %dth network (lowest loss)' % epoch)
    else:
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch), map_location=device)
        print('Loaded %dth network' % epoch)

    netG.load_state_dict(dict_net['net'])

    return netG, epoch

# Calculate the normalisation factors of the image stack
def calc_normfactors(x, pmin=0.3, pmax=98.5, axis=None):
    """Percentile-based image normalization."""

    x = x.copy()

    # Normalize the data 
    if x.dtype == 'float64':
        x = x.astype(np.int16)
        x = (x / (2 * np.iinfo(x.dtype).max)).astype(np.float64) + 0.5

    if x.dtype == 'int16':
        x = (x / (2 * np.iinfo(x.dtype).max)).astype(np.float64) + 0.5

    if x.dtype == 'uint16':
        x = (x / np.iinfo(x.dtype).max).astype(np.float64)

    if x.dtype == 'int8':
        x = (x / np.iinfo(x.dtype).max).astype(np.float64)

    if x.dtype == 'uint8':
        x = (x / np.iinfo(x.dtype).max).astype(np.float64)

    mi = np.percentile(x,pmin,axis=axis,keepdims=False)
    ma = np.percentile(x,pmax,axis=axis,keepdims=False)
    print(f"Minimum: {mi}\t Maximum: {ma}")
    return mi, ma

# Backconversion
class NormFloat2UInt16_round(object):
    
    def __init__(self, percent=0.8):
        self.percent = percent

    def __call__(self, data):

        data = (data * self.percent * np.iinfo(np.uint16).max)
        data = np.clip(data, np.iinfo(np.uint16).min, np.iinfo(np.uint16).max)
        return np.rint(data).astype(np.uint16)
    
class NormFloat2Int16_round(object):
    
    def __init__(self, percent=1.0):
        self.percent = percent

    def __call__(self, data):

        data = (data * self.percent - 0.5) * (np.iinfo(np.int16).max + np.abs(np.iinfo(np.int16).min))
        data = np.clip(data, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
        return np.rint(data).astype(np.int16)
    
# For the overlapping and weighting it should not be rounded
class NormFloat2UInt16(object):
    
    def __init__(self, percent=1.0):
        self.percent = percent

    def __call__(self, data):

        data = (data * self.percent * np.iinfo(np.uint16).max)
        data = np.clip(data, np.iinfo(np.uint16).min, np.iinfo(np.uint16).max)
        return data
    
class NormFloat2Int16(object):
    
    def __init__(self, percent=1.0):
        self.percent = percent

    def __call__(self, data):

        data = (data * self.percent - 0.5) * (np.iinfo(np.int16).max + np.abs(np.iinfo(np.int16).min))
        data = np.clip(data, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
        return data
    
class NormFloat2UInt8(object):
    
    def __init__(self, percent=1.0):
        self.percent = percent

    def __call__(self, data):

        data = (data * self.percent * np.iinfo(np.uint8).max)
        data = np.clip(data, np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
        return data
    
class UInt162NormFloat(object):

    def __call__(self, data):

        data = data / np.iinfo(np.uint16).max
        return data

## Helper function for the PerStruc-Algorithm

# 2D Autocorrelation
# Adapted from https://github.com/juglab/n2v/blob/master/n2v/utils/n2v_utils.py
def autocorrelation(image):
    """
    nD autocorrelation
    remove mean per-patch (not global GT)
    normalize stddev to 1
    value at zero shift normalized to 1...
    """

    # Get a smaller patch of the image
    x = image[image.shape[0]//2 - 10:image.shape[0]//2 + 10,
              image.shape[1]//2 - 10:image.shape[1]//2 + 10]

    x = (x - np.mean(x))/np.std(x)
    x  = np.fft.fftn(x)
    x  = np.abs(x)**2
    x = np.fft.ifftn(x).real
    x = x / x.flat[0]
    x = np.fft.fftshift(x)
    return x

# Line-wise FFT
def line_fft(image, should_fftshift=False):
    img_fft_line = np.empty(image.shape, np.complex128)
    for ind, line in enumerate(image):
        if should_fftshift:
            img_fft_line[ind] = fftshift(fft(line))
        else:
            img_fft_line[ind] = fft(line)
    return img_fft_line

# Line-wise mean power spectrum
def line_power_spectrum(image, should_norm=True):
    # For FFT the line power spectrum should have odd numbers of elements
    if  image.shape[1]%2==0:
        image = image[:, :-1]

    line_fft_img = line_fft(image, should_fftshift=True)
    line_spectrum = np.mean(np.abs(line_fft_img), axis=0)

    freqs = fftshift(fftfreq(len(line_spectrum)))

    # Since the main peak in the center is not interesting, just return the first half
    freqs_small = freqs[:len(freqs)//2]
    line_spectrum_small = line_spectrum[:len(line_spectrum)//2]

    if should_norm:
        min_val, max_val = np.min(line_spectrum_small), np.max(line_spectrum_small)
        line_spectrum_small =  (line_spectrum_small - min_val)/(max_val - min_val)
    
    return freqs_small, line_spectrum_small