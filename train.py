""" This module contains only the train method """

from network import Noise2NoiseUNet3D, init_weights
from dataset import N2VDataset3D
from transformations import *
import utils

import torch
import torch.nn as nn

# to store the data as tif file (here only the validation result are stored as tiff)
import tifffile

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

class Trainer:

    def __init__(self, data_dict):

        # Hyperparameters
        self.batch_size = data_dict['batch_size']
        self.num_epoch = data_dict['num_epoch']

        ## Adam optimizer
        self.lr = data_dict['lr']
        self.beta1 = data_dict['beta1']
        self.beta2 = data_dict['beta2']

        # Training state of the network
        self.train_continue = data_dict['train_continue']

        # Colormap for storing and showing the trained data
        self.cmap = data_dict['cmap']

        # Store and find - paths
        self.dir_checkpoint = data_dict['dir_checkpoint']
        self.dir_log = data_dict['dir_log']
        self.dir_result = data_dict['dir_result']
        self.dir_norm_factors = data_dict['dir_norm_factors']

        # Store & display frequency
        self.num_freq_disp = data_dict['num_freq_disp']
        self.num_freq_save = data_dict['num_freq_save']

        # Patch sizes (only single channel, since it is 3D Unet)
        self.ny = data_dict['ny']
        self.nx = data_dict['nx']
        self.nz = data_dict['nz']
        
        self.nch = data_dict['nch']
        
        # Augmentation data for the N2V augmenter
        self.perc_pixel = data_dict['perc_pixel']
        self.n2v_neighborhood_radius = data_dict['n2v_neighborhood_radius']
        self.structN2Vmask = data_dict['structN2Vmask']

        # Utilize the 3D Noise2Void2
        self.N2V2 = data_dict['N2V2']

        # Datasets as lists
        self.train_data_list = data_dict['train_dataset']
        self.val_data_list = data_dict['val_dataset']


        # check if we have  a gpu
        if torch.cuda.is_available():
            print("GPU is available")
            self.device = torch.device("cuda:0")
        else:
            print("GPU is not available")
            self.device = torch.device("cpu")

    # Store function
    def save(self, dir_chck, net, optim, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'net': net.state_dict(),
                    'optim': optim.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def save_best_model(self, dir_chck, netG, optimG, epoch, loss_G):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'net': netG.state_dict(),
                    'optim': optimG.state_dict(),
                    'epoch': epoch,
                    'loss': loss_G},
                   '%s/best_model.pth' % dir_chck)

    # Load function
    def load(self, dir_chck, net, optim=[], epoch=[]):

        if not os.path.exists(dir_chck) or not os.listdir(dir_chck):
            epoch = 0
            return net, optim, epoch

        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        net.load_state_dict(dict_net['net'])
        optim.load_state_dict(dict_net['optim'])

        return net, optim, epoch
    
    # Calculate the normfactors
    def calc_normfactors(self, x, pmin=0.3, pmax=98.5, axis=None):
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
    

    # Train function
    def train(self):

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr = self.lr

        batch_size = self.batch_size
        device = self.device

        size_inputstack = (self.nz, self.ny, self.nx)

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        train_data_list = self.train_data_list
        val_data_list = self.val_data_list

        # for the colormapping
        change_cm = utils.ChangeColormap4Tensorboard(neglect_alpha=True, cmap=self.cmap)
        batchtrafo = utils.BatchTransformer()

        # for conversion
        backconv = utils.NormFloat2UInt16_round()


        ## setup dataset

        ### Checkpoint folder
        dir_chck = self.dir_checkpoint

        ### Log folder
        dir_log_train = os.path.join(self.dir_log, 'train')
        dir_log_val = os.path.join(self.dir_log, 'val')

        ### Result folder
        dir_result_train = os.path.join(self.dir_result, 'train')
        dir_result_val = os.path.join(self.dir_result, 'val')
        if not os.path.exists(os.path.join(dir_result_train, 'images')):
            os.makedirs(os.path.join(dir_result_train, 'images'))
        if not os.path.exists(os.path.join(dir_result_val, 'images')):
            os.makedirs(os.path.join(dir_result_val, 'images'))

        ## Load normalization factors
        fname_norm_train = os.path.join(self.dir_norm_factors, 'train_dataset_min_max.txt')
        fname_norm_val = os.path.join(self.dir_norm_factors, 'validation_dataset_min_max.txt')
        if os.path.isfile(fname_norm_train):
            with open(fname_norm_train, 'r') as f:
                min_train = float(next(f))
                max_train = float(next(f))
                f.close()
        else:
            min_train, max_train = self.calc_normfactors(train_data_list[0])

        if os.path.isfile(fname_norm_val):
            with open(fname_norm_val, 'r') as f:
                min_val = float(next(f))
                max_val = float(next(f))
                f.close()
        else:
            min_val, max_val = self.calc_normfactors(val_data_list[0])

        ## Generate the transfromations

        ### Transformation
        transform_train = transforms.Compose([
            PercentileNormalize3D(mi=min_train, ma=max_train),
            RandomFlip3D(),
            RandomCrop3D((size_inputstack[1], size_inputstack[2])),
            N2V_mask_generator(size_inputstack,
                               nch=self.nch,
                               perc_pixel=self.perc_pixel,
                               n2v_neighborhood_radius=self.n2v_neighborhood_radius,
                               blindspot_strategy = "median",
                               structN2Vmask=self.structN2Vmask),
            ToTensor3D()
            ])
        transform_val = transforms.Compose([
            PercentileNormalize3D(mi=min_val, ma=max_val),
            RandomFlip3D(),
            RandomCrop3D((size_inputstack[1], size_inputstack[2])),
            N2V_mask_generator(size_inputstack,
                               nch=self.nch,
                               perc_pixel=self.perc_pixel,
                               n2v_neighborhood_radius=self.n2v_neighborhood_radius,
                               blindspot_strategy = "median",
                               structN2Vmask=self.structN2Vmask),
            ToTensor3D()
            ])

        ### Inverse Transformation
        transform_inv_train = transforms.Compose([
            ToNumpy3D(),
            PercentileDenormalize3D(mi=min_train, ma=max_train)
            ])
        transform_inv_val = transforms.Compose([
            ToNumpy3D(),
            PercentileDenormalize3D(mi=min_val, ma=max_val)
            ])

        ## Generate Datasets

        dataset_train = N2VDataset3D(
                train_data_list,
                stack_size = size_inputstack[0],
                transform=transform_train
                )
        dataset_val = N2VDataset3D(
                val_data_list,
                stack_size = size_inputstack[0],
                transform=transform_val
                )

        ## Generate DataLoaders

        loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0)
        loader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0)

        ## Information about the datasets for the training

        num_train = len(dataset_train)
        num_val = len(dataset_val)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
        num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))

        ## setup network
         
        net = Noise2NoiseUNet3D(in_channels = 1,
                                out_channels = 1,
                                is_N2V2_setup=self.N2V2,
                                final_sigmoid = False).to(device)

        init_weights(net, init_type='normal', init_gain=0.02)
        
        ## setup loss

        L1_pixelwise = nn.L1Loss().to(device)  # Regression loss: L1
        L2_pixelwise = nn.MSELoss().to(device)     # Regression loss: L2

        ## setup optimization
        params = net.parameters()

        optim = torch.optim.Adam(params, lr=lr, betas=(self.beta1, self.beta2))

        ## Load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            net, optim, st_epoch = self.load(dir_chck, net, optim)

        ## Setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)
        writer_val = SummaryWriter(log_dir=dir_log_val)

        ###########################################
        # make the model visible on tensorboard
        data_example = next(iter(loader_train))
        writer_train.add_graph(net, data_example['input'].to(device), verbose=False)
        ##########################################

        ## Training

        # Variable to check if it is the best loss
        loss_total_best = np.inf
        
        for epoch in range(st_epoch + 1, num_epoch + 1):

            ## training phase
            net.train()

            loss1_train = []
            loss2_train = []
            loss_total_train = []

            for batch, data in enumerate(loader_train, 1):

                def should(freq):
                    return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

                input, label, mask = data['input'].to(device), data['label'].to(device), data['mask'].to(device)

                # forward net
                output = net(input)

                # backward net
                optim.zero_grad()

                loss1 = L1_pixelwise(output * (1-mask), label * (1-mask))
                loss2 = L2_pixelwise(output * (1-mask), label * (1-mask))
                loss_total = 0.5*loss1 + 0.5*loss2
                
                loss_total.backward()
                optim.step()

                # get losses
                loss1_train += [loss1.item()]
                loss2_train += [loss2.item()]
                loss_total_train += [loss_total.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS1: %.4f: LOSS2: %.4f: LOSS_TOTAL: %.4f'
                      % (epoch, batch, num_batch_train, np.mean(loss1_train), np.mean(loss2_train), np.mean(loss_total_train)))

                if should(num_freq_disp):
                    ## show input, target and output
                    input = transform_inv_train(input)[0, (size_inputstack[0]//2)-8:(size_inputstack[0]//2)+8]
                    label = transform_inv_train(label)[0, (size_inputstack[0]//2)-8:(size_inputstack[0]//2)+8]
                    difference = np.clip(np.abs(input - label), 0, 1)
                    output = transform_inv_train(output)[0, (size_inputstack[0]//2)-8:(size_inputstack[0]//2)+8]

                    # change the colormapping
                    input_cm = batchtrafo(change_cm, input)
                    label_cm = batchtrafo(change_cm, label)
                    difference_cm = batchtrafo(change_cm, difference)
                    output_cm = batchtrafo(change_cm, output)

                    writer_train.add_images('input', input_cm, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                    writer_train.add_images('label', label_cm, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                    writer_train.add_images('difference (input-label)', difference_cm, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                    writer_train.add_images('output', output_cm, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

                    input = np.clip(input, 0, 1).squeeze()
                    label = np.clip(label, 0, 1).squeeze()
                    output = np.clip(output, 0, 1).squeeze()
                    dif = np.clip(abs(label - input), 0, 1).squeeze()

                    for j in range(label.shape[0]):
                        name = num_batch_train * (batch - 1) + j
                        fileset = {'name': name,
                                   'input': "%04d-input.png" % name,
                                   'output': "%04d-output.png" % name,
                                   'label': "%04d-label.png" % name,
                                   'dif': "%04d-dif.png" % name}

                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['input']), input[j, :, :], cmap=self.cmap)
                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['output']), output[j, :, :], cmap=self.cmap)
                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['label']), label[j, :, :], cmap=self.cmap)
                        plt.imsave(os.path.join(dir_result_train, 'images', fileset['dif']), dif[j, :, :], cmap=self.cmap)

            writer_train.add_scalar('loss1', np.mean(loss1_train), epoch)
            writer_train.add_scalar('loss2', np.mean(loss2_train), epoch)
            writer_train.add_scalar('loss_totoal', np.mean(loss_total_train), epoch)

            ## validation phase
            with torch.no_grad():
                net.eval()

                loss1_val = []
                loss2_val = []
                loss_total_val = []

                for batch, data in enumerate(loader_val, 1):
                    def should(freq):
                        return freq > 0 and (batch % freq == 0 or batch == num_batch_val)

                    input, label, mask = data['input'].to(device), data['label'].to(device), data['mask'].to(device)

                    # forward net
                    output = net(input)

                    # Calculate the loss
                    loss1 = L1_pixelwise(output * (1-mask), label * (1-mask))
                    loss2 = L2_pixelwise(output * (1-mask), label * (1-mask))
                    loss_total = 0.5*loss1 + 0.5*loss2

                    loss1_val += [loss1.item()]
                    loss2_val += [loss2.item()]
                    loss_total_val += [loss_total.item()]

                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS1: %.4f: LOSS2: %.4f: LOSS_TOTAL: %.4f'
                          % (epoch, batch, num_batch_val, np.mean(loss1_val), np.mean(loss2_val), np.mean(loss_total_val)))

                    if should(num_freq_disp):
                        ## show input, target and output
                        input = transform_inv_train(input)[0, (size_inputstack[0]//2)-8:(size_inputstack[0]//2)+8]
                        label = transform_inv_train(label)[0, (size_inputstack[0]//2)-8:(size_inputstack[0]//2)+8]
                        difference = np.clip(np.abs(input - label), 0, 1)
                        output = transform_inv_train(output)[0, (size_inputstack[0]//2)-8:(size_inputstack[0]//2)+8]

                        # change the colormapping
                        input_cm = batchtrafo(change_cm, input)
                        label_cm = batchtrafo(change_cm, label)
                        difference_cm = batchtrafo(change_cm, difference)
                        output_cm = batchtrafo(change_cm, output)

                        writer_val.add_images('input', input_cm, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                        writer_val.add_images('label', label_cm, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                        writer_val.add_images('difference', difference_cm, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                        writer_val.add_images('output', output_cm, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

                        input = np.clip(input, 0, 1).squeeze()
                        label = np.clip(label, 0, 1).squeeze()
                        output = np.clip(output, 0, 1).squeeze()
                        dif = np.clip(abs(label - input), 0, 1).squeeze()

                        # change the range of the data, so that it has uint16/int16-format
                        input_conv = batchtrafo(backconv, input)
                        label_conv = batchtrafo(backconv, label)
                        output_conv = batchtrafo(backconv, output)
                        dif_conv = batchtrafo(backconv, dif)


                        for j in range(label.shape[0]):
                            name = num_batch_val * (batch - 1) + j
                            fileset = {'name': name,
                                       'input': "%04d-input.tif" % name,
                                       'output': "%04d-output.tif" % name,
                                       'label': "%04d-label.tif" % name,
                                       'dif': "%04d-dif.tif" % name}

                            tifffile.imsave(os.path.join(dir_result_val, 'images', fileset['input']), input_conv[j, :, :])
                            tifffile.imsave(os.path.join(dir_result_val, 'images', fileset['output']), output_conv[j, :, :])
                            tifffile.imsave(os.path.join(dir_result_val, 'images', fileset['label']), label_conv[j, :, :])
                            tifffile.imsave(os.path.join(dir_result_val, 'images', fileset['dif']), dif_conv[j, :, :])

                writer_val.add_scalar('loss1', np.mean(loss1_val), epoch)
                writer_val.add_scalar('loss2', np.mean(loss2_val), epoch)
                writer_val.add_scalar('loss_totoal', np.mean(loss_total_val), epoch)

            loss_total_train_mean = np.mean(loss_total_train)
            if (loss_total_best > loss_total_train_mean) and (epoch > 0.25*num_epoch):
                loss_total_best = loss_total_train_mean
                self.save_best_model(dir_chck, net, optim, epoch, loss_total_train_mean)
            
            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, net, optim, epoch)

        writer_train.close()
        writer_val.close()
