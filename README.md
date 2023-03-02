# Deep3P-Denoising
Deep3P Denoising: Code of the 3PM-Noise2Void and PerStruc-Denoiser Method

ADD SMALL SUMMARY OF PROJECT AND RESULTING DATA

This Matlab code is part of the improvements made to the neuroscientific, open-source software **CaSCaDe 2.0** (**Ca**lcium **S**ignal Analysis and **Ca**lcium **De**coding **2.0**). It replaces the classical noise filtering of the previous version of [CaSCaDe](https://pubmed.ncbi.nlm.nih.gov/28132831/) [[1]](#references) with unsupervised deep-learning based denoising (**3D-Noise2Noise** [[2]](#references)[[3]](#references)). The code helps to remove noise from calcium signals that were hidden by the inherent 2-photon and 3-photon microscopy noise, allowing the segmentation and detection pipeline to work more accurately and robustly.
These Python scripts contain the following functionality based on the 3D-Noise2Noise method:

- [**Training:**](#training) Optimized training on selected data.
- [**Inference:**](#inference) Optimized denoising of given data with selected, trained model.
- [**Transfer-Learning:**](#transfer-learning) Optimized adjustment of a pre-trained model to new selected data.

|  ![Results of the 3DN2N restored 2P data](docs/images/2P/figure_2P.gif) |
|:---|
|**Deep-learning restored two photon microscopy recording:**
**(a)** An overlay of raw and denoised calcium signals of astrocytes.
**(b)** Zoomed-in views on a region of interest, showcasing the raw and the denoised recording in finer detail.|

## Table of Contents

- [Deep3P-Denoising](#deep3p-denoising)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [How-To-Run](#how-to-run)
    - [Training](#training)
    - [Inference](#inference)
    - [Transfer-Learning](#transfer-learning)
    - [Further information](#further-information)
  - [Future Work](#future-work)
  - [References](#references)
  - [Acknowledgements](#acknowledgements)

## Dataset

- **Description:** The data used in this research project consisted of in-vivo calcium signals from astrocytes in the deep layers of the mouse brain, recorded using two-photon and three-photon microscopy. These data are videos of the same x-y-plane showing the temporal intensity changes of the GCaMP-labeled calcium signals.
- **Data requirement:** The duration of brightness changes should not happen faster than twice the frame rate. This constraint is not necessary for inference, as the entire recording is used.
- **Preprocessing:** Before the video data can be used for training or inference, it must be preprocessed. This includes registration, alignment, and cropping of translational borders.

## Dependencies

To run the code in these projects, you will need to have the following dependencies installed:

- `Python 2.7`
- `Pytorch 1.0.0+`
- `torchvision 0.2.1+`
- `numpy`
- `PIL`
- `os`
- `TensorFlow 1.1.0+` (for tensorboard)
- `Linux`


## Installation

To install and use these denoising scripts, follow these steps:

- Clone the repository using the following command in your terminal:
```sh
git clone https://gitlab.com/suamE/cascadedenoisingstrategies.git
```
- Navigate to the cloned repository by using the following command:
```sh
cd cascadedenoisingstrategies
```
- Launch Matlab by typing ``matlab`` in your terminal.
- Add the *utils* folder and its subfolders to Matlab's search path by entering the following command in Matlab's command prompt:
```Matlab
addpath(genpath("utils"));
```

## How-To-Run


There are three main scripts for **training** ([``training_routine.m``](training_routine.m)), **inference**([``prediction_3D_overlapping.m``](prediction_3D_overlapping.m)), and **transfer learning**([``transfer_learning_routine.m``](transfer_learning_routine.m)).

Below is a description of how to run each script:

### Training
```Matlab
[trained_model, denoised_img_stack_cell] = training_routine();
```
- **Instructions:**
	1. Select the image stacks to use for training. If no further selection is needed, click cancel.
	2. Select the saving path of the trained model and choose its name.

- **Training process:**
	- Two diagrams will be displayed, showing the change in the average loss (training and validation loss) per training iteration.

	|  ![Loss diagrams](docs/images/logger/logger_loss.PNG) |
	|:---|
	|**Normalized loss values per iteration:** The normalized and averaged training loss (left diagram) is automatically updated each batch iteration, whereas the validation loss (right diagram) is updated each epoch.|

	- Two figures will be displayed, showing the performance of the trained model on example image patches, updated each epoch.

	|  ![Example images](docs/images/logger/logger_images.PNG) |
	|:---|
	|**Examples of input, target and prediction patches per epoch:** Different examples of input, target, and prediction patches are shown in each row. One figure (left) shows samples from the training dataset and the other (right) shows samples from the validation dataset, providing a qualitative impression of the training results.|

- **Outputs:**
	- ``trained_model``: a ``dlnetwork`` trained on the data
	- ~~``denoised_img_stack_cell``: a ``cell array`` containing the denoised data~~ (the current version returns no prediction due to a bug in the prediction pipeline)

### Inference
```Matlab
denoised_img_stack_cell = prediction_3D_overlapping();
```
- **Instructions:**
	1. Select the image stacks to which the trained model should be applied. If no further selection is needed, click cancel.
	2. Select the trained model to use for denoising the new data.
	3. Select the saving path in which to store the denoised data.

- **Outputs:**
	- ``denoised_img_stack_cell``: a ``cell array`` containing the denoised data


### Transfer-Learning
```Matlab
[trained_model, denoised_img_stack_cell] = transfer_learning_routine();
```
- **Instructions:**
	1. Select the image stacks on which the trained model should be adapted. If no further selection is needed, click cancel.
	2. Select the trained model to be adapted to the new data.
	3. Select the saving path for the retrained model and choose its name.

- **Transfer-Learning process:**
	- Two diagrams will be displayed, showing the change in average loss (training and validation loss) per training iteration (similar to the [training process](#training)).
	- Two figures will displayed, showing the performance of the trained model on example image patches. These figures will be automatically updated each epoch (similar to the [training process](#training)).

- **Outputs:**
	- ``trained_model``: a ``dlnetwork`` trained on the data
	- ~~``denoised_img_stack_cell``: a ``cell array`` containing the denoised data~~ (the current version returns no prediction due to a bug in the prediction pipeline)

### Further information

For more detailed information about the code and its functionality, please refer to the [documentation file](docs/documentation/Documentation.txt) and the code itself, which includes comments explaining the various functions and steps. Additionally, you may contact the authors for further assistance or clarification.

## Future Work

- [X] Document the code
- [ ] Fix certain bugs
	- [ ] Make predictions directly after training in	    [training_routine.m](https://gitlab.com/suamE/cascadedenoisingstrategies/-/blob/main/training_routine.m#L169).
- [ ] Implement certain features
	- [ ] Implement/Add a function, which allows for predictions on a test substack instead of random patches in [training_routine](training_routine.m) and [transfer_learning_routine](transfer_learning_routine.m).
	- [ ] Implement a function, that adjusts the input dimension/ patch size based on stored prediction parameters, which corresponds to the "conservation of volume" (e.g. if 16x32x32x32 fits on the GPU, then 1x64x64x64 should also be able to run).

## References

1. Amit Agarwal et al., “Transient opening of the mitochondrial permeability transition pore induces microdomain calcium transients in astrocyte processes”, In: Neuron 2017
2. Jaakko Lehtinen et al., “Noise2Noise: Learning Image Restoration without Clean Data”, In: arXiv preprint 2018
3. Xinyang Li et al., “Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising”, In: Nat. Methods 2021

## Acknowledgements

This code was developed and implemented by Emanuel Maus as part of the **CaSCaDe 2.0** (**Ca**lcium **S**ignal Analysis and **Ca**lcium **De**coding **2.0**) project, with guidance and consultation of [Amit Agarwal](https://www.uni-heidelberg.de/izn/researchgroups/agarwal/), [Khaleel Alhalaseh](https://www.researchgate.net/profile/Khaleel-Alhalaseh) and [Robert Prevedel](https://www.prevedel.embl.de/). If you have any questions, suggestions, or have noticed any bugs, please feel free to contact me at emanuel.maus AT embl DOT de.