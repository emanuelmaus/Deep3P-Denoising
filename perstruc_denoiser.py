""" Contains the code for the Perstruc-Denoiser algorithm """ 

# Imports 
import numpy as np
import copy
from tqdm import tqdm

from scipy.fftpack import fft, ifft
from scipy.ndimage import gaussian_filter1d

def perstruc_denoiser(img_stack, FREQ_POS=72, use_gaussian=False):

    # Output image stack
    modified_img_stack = np.empty(img_stack.shape, dtype=img_stack.dtype)
    # Median value of the original image stack
    median_value = np.median(img_stack)
    # Single reference line for the whole image stack
    reference_fft_line = None
    bool_fft_reference_set = False

    for img_ind, img in enumerate(tqdm(img_stack)):
        # Copy of the original image for further processing
        img_cp = copy.deepcopy(img)
        img_cp_acqu = copy.deepcopy(img)

        # Reverse each line
        img_cp_acqu[1::2] = img_cp_acqu[1::2][:, ::-1]

        # Line-wise fourier transformation
        img_fft_line = np.empty(img_cp_acqu.shape, np.complex)
        for ind, line in enumerate(img_cp_acqu):
            img_fft_line[ind] = fft(line)

        # Determine reference line
        lines_std = np.std(img_cp_acqu, axis=1)
        ref_row_no = np.argmin(lines_std)
        # Frequency of the periodic structured noise
        ref_col_no = FREQ_POS

        # Store the reference line, if None has been identified
        if reference_fft_line is None:
            reference_fft_line = img_fft_line[ref_row_no, ref_col_no].copy()
        # Determine the phase shift in integer
        phases = np.angle(img_fft_line[:, ref_col_no]/reference_fft_line)
        phases_int = np.rint(phases).astype(np.int8)

        # Shift the rows except the reference line
        img_cp_shifted = copy.deepcopy(img_cp_acqu)
        for ind, line in enumerate(img_cp_shifted):
            if (ind != ref_row_no) and bool_fft_reference_set:
                img_cp_shifted[ind] = np.roll(line, 1*phases_int[ind])

        bool_fft_reference_set = True
        
        # Calculate the pattern of the periodic signal and subtract it from the image
        median_line = np.median(img_cp_shifted, axis=0)
        img_cp_shifted_recon = img_cp_shifted - median_line[None]

        # Reverse the shifting and the flipping 
        img_cp_recon = np.empty(img_cp_shifted_recon.shape)
        for ind, line in enumerate(img_cp_shifted_recon):
            if ind != ref_row_no:
                img_cp_shifted_recon[ind] = np.roll(line, -1*phases_int[ind])

        img_cp_recon = img_cp_shifted_recon
        img_cp_recon[1::2] = img_cp_recon[1::2][:, ::-1]

        img_cp = img_cp_recon
        modified_img_stack[img_ind] = np.rint(img_cp_recon).astype(img_stack.dtype)
    
    # Since the median was subtracted, we need to add it afterwards
    median_value_mod = np.median(modified_img_stack)
    modified_img_stack = modified_img_stack + median_value - median_value_mod

    # In case of high frequency periodic structured noise, apply a 1D Gaussian filter
    if use_gaussian:
        modified_img_stack = gaussian_filter1d(modified_img_stack)

    return np.rint(modified_img_stack).astype(img_stack.dtype)

