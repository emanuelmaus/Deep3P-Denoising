import numpy as np
import numpy.ma as ma
import torch

## Dataset for training
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

## Dataset for inference

class InferenceDataset3D(torch.utils.data.Dataset):
    """
    dataset of a list of image stacks with the following characteristics
        - different lateral image sizes
        - stack size differ
    """
    
    def __init__(self, data, z_crop_width, z_inputstack, stride_to_size_ratio, transform=None):

        self.data = data
        self.front_width, self.tail_width = z_crop_width 
        self.z_inputstack = z_inputstack
        self.stride_to_size_ratio = stride_to_size_ratio
        self.transform = transform
        
        self.original_size = self.data.shape
        self.original_z_dim = self.original_size[0]
        
        self.new_z_dim, self.new_z_inputstack, self.stride = self._generate_parameters()
        self.index_array = self._generate_index_array()
        self.norm_factors = self._generate_norm_factors()
        self.pad_extension_front, self.pad_extension_tail = self._generate_pad_extensions()
        self.data = self._extend_data()
       

    def __getitem__(self, index):


        current_index = self.index_array[index]
        cropped_current_index = current_index[self.front_width:-self.tail_width]
        # Shift the index for the front_width size
        cropped_current_index = cropped_current_index - self.front_width
        # Since we only consider the non cropped weights, our norm_factor has - front_width element less
        cropped_norm_factors = self.norm_factors[cropped_current_index]
        input = self.data[current_index][..., None]
        

        if input.dtype == 'float64':
            input = input.astype(np.int16)
            
            input = (input / (2 * np.iinfo(input.dtype).max)).astype(np.float64) + 0.5

        if input.dtype == 'int16':
            input = (input / (2 * np.iinfo(input.dtype).max)).astype(np.float64) + 0.5

        if input.dtype == 'uint16':
            input = (input / np.iinfo(input.dtype).max).astype(np.float64)
            
        if input.dtype == 'int8':
            input = (input / np.iinfo(input.dtype).max).astype(np.float64)

        if input.dtype == 'uint8':
            input = (input / np.iinfo(input.dtype).max).astype(np.float64)
            
        if input.ndim == 2:
            input = np.expand_dims(input, axis=2)

        if self.transform:
            data = self.transform(input)
        else:
            data = input

        return data, cropped_current_index, cropped_norm_factors

    def __len__(self):
        return int(len(self.index_array))

     
    def _generate_parameters(self):
        z_inputstack_front_tail = self.front_width + self.z_inputstack + self.tail_width 
        z_dim_front_tail = self.front_width + self.original_z_dim + self.tail_width
        # Modify, if the stack_size%stride_to_size_ratio=!=0
        new_z_inputstack = z_inputstack_front_tail + self.z_inputstack%self.stride_to_size_ratio

        # Stride needed to be adjusted by - 1
        stride = int(new_z_inputstack/self.stride_to_size_ratio)-1

        new_z_dim = z_dim_front_tail + z_dim_front_tail%stride

        return new_z_dim, new_z_inputstack, stride

    
    def _generate_index_array(self):
        n_h = self.new_z_dim - self.stride
        return np.arange(self.new_z_inputstack)[None, :] + np.arange(0, n_h, self.stride)[:, None]
    
    def _generate_pad_extensions(self):
        return self.front_width, self.index_array.max() - self.original_z_dim - self.front_width + 1 
        
    def _extend_data(self):
        return np.pad(self.data, ((self.pad_extension_front, self.pad_extension_tail),(0,0),(0,0)), mode='symmetric') 
    
    def _generate_norm_factors(self):
        _, counts = np.unique(self.index_array[:, self.front_width:-self.tail_width], return_counts=True)
        return 1/counts
    
    def get_pad_extensions(self):
        
         return self.pad_extension_front, self.pad_extension_tail
    
    def get_index_array(self):
        
        return  self.index_array
    
    def get_cropping_indices(self):
        
        return 0, self.original_z_dim
    
    def get_output_size(self):
       
        return (self.original_z_dim + self.pad_extension_tail - self.tail_width, self.original_size[1], self.original_size[2])