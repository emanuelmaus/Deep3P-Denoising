https://gitlab.com/suamE/cascadedenoisingstrategies.git# Deep3P-Denoising
Deep3P Denoising: Code of the 3PM-Noise2Void and PerStruc-Denoiser Method

ADD SMALL SUMMARY OF PROJECT AND RESULTING DATA
## Table of Contents

- [Table of Contents](#table-of-contents)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [How-To-Run](#how-to-run)
  - [Training](#training)
  - [Inference](#inference)
  - [PerStruc-Denoiser](#perstruc-denoiser)
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
git clone https://github.com/emanuelmaus/Deep3P-Denoising.git
```
- Navigate to the cloned repository by using the following command:
```sh
cd Deep3P-Denoising/notebooks
```
- Launch Jupyter-lab by typing ``jupyter-lab`` in your terminal.

## How-To-Run


There are three notebooks for **training** ([``01_Training-3PM-Noise2Void.ipynb``](/notebooks/01_Training-3PM-Noise2Void.ipyn)), **inference**([``02_Inference-3PM-Noise2Void.ipynb``](/notebooks/02_Inference-3PM-Noise2Void.ipynb)), and **PerStruc-Denoiser**([``03_PerStruc-Denoiser.ipynb``](/notebooks/03_PerStruc-Denoiser.ipynb)).

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


### PerStruc-Denoiser
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

For more detailed information about the code and its functionality, please refer to the [documentation file](/docs/documentation/Documentation.txt) and the code itself, which includes comments explaining the various functions and steps. Additionally, you may contact the authors for further assistance or clarification.

## Future Work

- [X] Document the code
- [ ] Fix certain bugs
	- [ ] Make predictions directly after training in
- [ ] Implement certain features
	- [ ] Implement/Add a function, which allows for predictions on a test substack instead of random patches in.
	- [ ] Implement a function, that adjusts the input dimension/ patch size based on stored prediction parameters, which corresponds to the "conservation of volume" (e.g. if 16x32x32x32 fits on the GPU, then 1x64x64x64 should also be able to run).

## References

1. Amit Agarwal et al., “Transient opening of the mitochondrial permeability transition pore induces microdomain calcium transients in astrocyte processes”, In: Neuron 2017
2. Jaakko Lehtinen et al., “Noise2Noise: Learning Image Restoration without Clean Data”, In: arXiv preprint 2018
3. Xinyang Li et al., “Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising”, In: Nat. Methods 2021

## Acknowledgements

This code was developed and implemented by Emanuel Maus as part of the **CaSCaDe 2.0** (**Ca**lcium **S**ignal Analysis and **Ca**lcium **De**coding **2.0**) project, with guidance and consultation of [Robert Prevedel](https://www.prevedel.embl.de/) et al.. If you have any questions, suggestions, or have noticed any bugs, please feel free to contact me at emanuel.maus AT embl DOT de.