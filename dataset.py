import numpy as np
import numpy.ma as ma
import torch
from skimage import transform
import os
import copy
import tifffile

class N2VDataset3D(torch.utils.data.Dataset):
    """
    dataset of a list of image stacks with the following characteristics
        - different lateral image sizes
        - stack size differ
    """

    def __init__(self, data_list, stack_size, transform=None):

        self.data_list = data_list

        self.stack_size = stack_size
        
        # Extend the stack size, so that it fits into the 3D network
        self.__extend__()

        self.transform = transform

        self.no_stacks = len(self.data_list)
        self.no_imgs_each_stack = np.array([len(stack) for stack in self.data_list])

        self.index_patches = self.__extract_patches__()
        # Extract patches of image stack
        self.index_each_stack = np.array([len(index_stack) for index_stack in self.index_patches])
        

    def __getitem__(self, index):

        # Since the number of images per stack differ and also its size, it is needed to find out which stack the index is pointing to
        # The first negative entry is the stack which is point to
        temp_index_array = index - self.index_each_stack.cumsum()
        stack_no = int(np.where(temp_index_array< 0)[0][0])
        # The image index is the absolute number of the temp_index_array entry at stack_no
        index_ind = int(temp_index_array[stack_no] + self.index_each_stack[stack_no])

        index_img_stack = self.index_patches[stack_no][index_ind]
        sample_img_stack = self.data_list[stack_no][index_img_stack]

        # The data is stored as float64, if it was translated for compensation of the shift, but contains only integer values
        # --> the data has to be converted to int16 and is normalized afterwards
        # Otherwise the data is often stored as int16 or uint16
        data = sample_img_stack[..., None]

        if data.dtype == 'float64':
            data = data.astype(np.int16)
            data = (data / (2 * np.iinfo(data.dtype).max)).astype(np.float64) + 0.5

        if data.dtype == 'int16':
            data = (data / (2 * np.iinfo(data.dtype).max)).astype(np.float64) + 0.5

        if data.dtype == 'uint16':
            data = (data / np.iinfo(data.dtype).max).astype(np.float64)

        if data.dtype == 'int8':
            data = (data / np.iinfo(data.dtype).max).astype(np.float64)
            
        if data.dtype == 'uint8':
            data = (data / np.iinfo(data.dtype).max).astype(np.float64)

        if data.ndim == 3:
            data = np.expand_dims(data, axis=3)

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return int(np.sum(self.index_each_stack))


    def __extend__(self):
        for index,stack in enumerate(self.data_list):
            temp_stack_size = stack.shape[0]
            # Since we need for this N2N method targets, we need to consider
            # twice the size the input_size
            additional_imgs = self.stack_size - (temp_stack_size% self.stack_size)
            #print(additional_imgs)
            self.data_list[index] = np.pad(stack, ((int(np.ceil(additional_imgs/2)),
                int(np.floor(additional_imgs/2))),(0,0), (0,0)), mode='symmetric')

    def __extract_patches__(self):
        patch_data_list = []
        for img_stack_size in self.no_imgs_each_stack:
            patch_data_list.append(np.arange(self.stack_size)[None, :] + np.arange((img_stack_size - 1) - (self.stack_size - 1))[:, None])
        return patch_data_list