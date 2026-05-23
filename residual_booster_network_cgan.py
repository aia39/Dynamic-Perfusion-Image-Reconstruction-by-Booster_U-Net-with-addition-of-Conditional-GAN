#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 02:34:16 2026

@author: maistiak
"""

import sys
sys.path.append('loss_functions')
sys.path.append('template_models/')
sys.path.append('utils')

import os.path
import torch.nn
import numpy as np
import torch
import time
import pickle
from datetime import datetime
from torchsummary import summary
from loadData import *  ##self-gated

from complex_pl_l1_loss import *
from booster_runet import *
from torch.utils.tensorboard import SummaryWriter
from supportingFunctions import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI)
import matplotlib.pyplot as plt
from tqdm import tqdm


def print_loss(losses, plot_name):
    l1_loss = []
    perceptual_loss = []
    total_loss = []
    val_loss = []
    for sublist in losses:
        total_loss.append(sublist[0].item())
        val_loss.append(sublist[1].item())
        l1_loss.append(sublist[2].item())
        perceptual_loss.append(sublist[3].item())
        
    epochs = np.arange(1, len(losses)+1)
    
    plt.figure()
    plt.plot(epochs, total_loss, label="Train Loss")
    plt.plot(epochs, perceptual_loss, label="Perceptual Loss")
    plt.plot(epochs, l1_loss, label="L1 Loss")
    plt.plot(epochs, val_loss, label="Valid Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losse Curve")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(plot_name)
    plt.close()



## 3D version

def conv3d_block(in_ch, out_ch, stride=(2,2,2), normalize=True):
    """
    Single discriminator block.
    stride can be a tuple to treat spatial and temporal dims differently.
    e.g. stride=(2,2,1) — downsample H/W but not T.
    """
    layers = [
        nn.Conv3d(
            in_ch, out_ch,
            kernel_size=4,
            stride=stride,
            padding=1,
            bias=False
        )
    ]
    if normalize:
        layers.append(nn.InstanceNorm3d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchGAN3DDiscriminator(nn.Module):
    """
    3D conditional PatchGAN for dynamic MRI.

    Input : cat(undersampled, recon_or_gt) → [B, 4, H, W, T]
    Output: patch-level real/fake map      → [B, 1, pH, pW, pT]

    spatial_stride  : (H, W) downsampling per layer
    temporal_stride : T downsampling per layer — set to 1 to be
                      conservative when T is small (e.g. T < 16)
    """
    def __init__(
        self,
        in_channels=2,          # 2 (undersampled) + 2 (recon/GT)
        base_filters=64,
        spatial_stride=2,
        temporal_stride=2,      # set to 1 if T is small (< 16 frames)
    ):
        super().__init__()

        s  = spatial_stride
        ts = temporal_stride
        f  = base_filters

        self.model = nn.Sequential(
            # Layer 1 — no norm on first layer (pix2pix convention)
            conv3d_block(in_channels, f,    stride=(s, s, ts),  normalize=False),
            # Layer 2
            conv3d_block(f,           f*2,  stride=(s, s, ts)),
            # Layer 3
            conv3d_block(f*2,         f*4,  stride=(s, s, ts)),
            # Layer 4 — stride=1 to preserve patch resolution before output
            conv3d_block(f*4,         f*8,  stride=(1, 1, 1)),
            # Output — single channel patch map
            nn.Conv3d(f*8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        """x: [B, 4, H, W, T]"""
        return self.model(x)


class GANLoss(nn.Module):
    """LSGAN (MSE) — more stable than vanilla BCE for medical imaging."""
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def __call__(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)
    

def build_optimizers(generator, discriminator, lr=2e-4, betas=(0.5, 0.999)):  #2e-4
    opt_G = torch.optim.Adam(generator.parameters(),     lr=lr, betas=betas)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    
    sch_G = ReduceLROnPlateau(opt_G, mode='min', factor=0.1, patience=4)
    sch_D = ReduceLROnPlateau(opt_D, mode='min', factor=0.1, patience=4)
    return opt_G, opt_D, sch_G, sch_D



def residual_booster_network(N,batch,in_features,out_features,resolution,layers,dp,alpha,beta,lr,nepoch,save_interval,directory,save_directory,pretrain,device,plot_name):

# -------------------------------------------------------------------------------------------
#
#     residual_booster_network(N,batch,in_features,out_features,resolution,layers,dp,alpha,beta,lr,nepoch,save_interval,directory,save_directory,device)
#
# -------------------------------------------------------------------------------------------
#    
#     inputs (2D radial SMS myocardial perfusion datasets)
#
#        -N (integer: 8|16|32): number of training time frames for each batch [default: 32]
#        -batch (integer): mini-batch size for training [default: 3]
#        -in_features (integer: 2): number of input channels (real and imaginary components)
#        -out_features (integer: 2): number of output channels (real and imaginary components)
#        -resolution (integer): number of filters in the first layer of each Unet in the residual booster network, doubled for each layer [default: 64]
#        -layers (integer): number of layers in each Unet in the residual booster network [default: 3]
#        -dp (boolean): adds 25% dropout and additional convolutional layers to each Unet [default: True]
#        -alpha (float): weight for perceptual loss component of the loss function [default: 0.04]
#        -beta (float): weight for the L1 component of the loss function [default: 1.0] 
#        -lr (float): learning rate for the adam optimizer [default: 0.0003]
#        -nepoch (integer): number of epochs to train the network [default: 100]
#        -save_interval (integer): epoch interval to save intermediate networks [default: 20]
#        -directory (string): directory to save the network and other parameters
#        -save_directory (string): name to save network
#        -device (cuda|cpu): determines whether to train network on GPU or CPU if not available
#        -pretrain (string): location of gated network for transfer learning

    #load SMS data for training
    train_set = loadData(N=N,split='train',do_transform=False,directory=directory,reduced_rays=True)
    validation_set = loadData(N=N,split='val',do_transform=False,directory=directory,reduced_rays=True)
    
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch,shuffle=True)    ##
    validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=batch,shuffle=False)
    
    #residual booster network initialization
    generator = booster_runet(in_features=in_features, out_features=out_features, resolution=resolution, layers=layers, dp=dp).to(device)   
    
    discriminator = PatchGAN3DDiscriminator(in_channels=2).to(device)
    
    
    if pretrain is not None:
        state = torch.load(pretrain)
        generator.load_state_dict(state['state_dict'])
    else:
        #orthogonal weight initialization
        generator.apply(generator.initialize_weights)

    opt_G, opt_D, sch_G, sch_D = build_optimizers(generator, discriminator)
    criterion_GAN = GANLoss()
    ##criterion_L1  = nn.L1Loss()
    
    pl_l1 = complex_pl_l1_loss(alpha=alpha,beta=beta)

    

    with open(directory+'scale_factors.pkl','rb') as f:
        mean,std = pickle.load(f)

    #tensorboard training visualization
    train_writer = SummaryWriter('runs/' + directory[15:] + 'training')
    validation_writer = SummaryWriter('runs/' + directory[15:] + 'validation')

    #tracker for calculating SSIM, PSNR, and NRMSE during training
    train_metric_tracker = metric_tracker(mean,std)
    validation_metric_tracker = metric_tracker(mean,std)
    
    print('beginning training...')
    print('network saved to ',directory,' as ', save_directory)

    step = 0
    currentLoss = 0
    all_loss = []
    all_loss_D = []
    all_loss_G = []
    pretrain_epochs = 0
    num_epochs=60
    lambda_adv=1.0
    
    for epoch in tqdm(range(nepoch)):
        start = time.time()
        for inputs,targets in train_loader:
            #optimizer.zero_grad()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            inputs = inputs.float()
            targets = targets.float()
            
            outputs = generator(inputs)
            
            real_pair_re = torch.stack([inputs[:,0,...], targets[:,0,...]], dim=1)  # [B, 2, H, W]
            real_pair_im = torch.stack([inputs[:,1,...], targets[:,1,...]], dim=1)  # [B, 2, H, W]
            
            fake_pair_re = torch.stack([inputs[:,0,...], outputs[:,0,...]], dim=1)  # [B, 2, H, W]
            fake_pair_im = torch.stack([inputs[:,1,...], outputs[:,1,...]], dim=1)  # [B, 2, H, W]
            
            
            # ── Step 1: Discriminator
            if epoch >= pretrain_epochs:
                opt_D.zero_grad()
                loss_D = 0.25 * (criterion_GAN(discriminator(real_pair_re),target_is_real=True) + \
                                criterion_GAN(discriminator(real_pair_im),target_is_real=True) + \
                    criterion_GAN(discriminator(fake_pair_re.detach()), target_is_real=False) + \
                    criterion_GAN(discriminator(fake_pair_im.detach()), target_is_real=False))
                
                loss_D.backward()
                opt_D.step()
            
            all_loss_D.append(loss_D.item())
            # ── Step 2: Generator ────────────────────────────────────────
            opt_G.zero_grad()

            # Adversarial
            if epoch >= pretrain_epochs:
                loss_adv = 0.5*(criterion_GAN(discriminator(fake_pair_re), target_is_real=True) + \
                    criterion_GAN(discriminator(fake_pair_im), target_is_real=True))
            else:
                loss_adv = torch.tensor(0.0, device=device)

            # Reconstruction (L1 + magnitude L1 + perceptual)
            loss_recon, ploss, l1loss = pl_l1(outputs,targets, verbose = False)
            ##loss_recon, loss_components = criterion_recon(recon, ground_truth)

            loss_G = lambda_adv * loss_adv + loss_recon
            loss_G.backward()
            opt_G.step()
            all_loss_G.append(loss_G.item())
            
            

            train_metric_tracker.calculate_metrics(outputs.detach().cpu().numpy(),targets.detach().cpu().numpy(),magnitude=False)

       
        if(epoch%10==0):
            plt.plot(all_loss_D,label='Discriminator loss')
            plt.plot(all_loss_G,label='Generator loss')
            plt.legend()
            plt.show()
        
        train_loss_dict = pl_l1.return_loss()
        train_metric_dict = train_metric_tracker.return_metrics()
        
        #tensorboard writer for training intermediate reconstructions, loss, and image metrics
        tb_add_scalar(train_writer,step,train_loss_dict)
        tb_add_scalar(train_writer,step,train_metric_dict)

        pl_l1.clear_loss()
        train_metric_tracker.clear_metrics()

        validate_network(generator,validation_loader,pl_l1,validation_metric_tracker,device)
        
        validation_loss_dict = pl_l1.return_loss()
        validation_metric_dict = validation_metric_tracker.return_metrics()

        #tensorboard writer for validation intermediate reconstructions, loss, and image metrics
        tb_add_scalar(validation_writer,step,validation_loss_dict)
        tb_add_scalar(validation_writer,step,validation_metric_dict)

        pl_l1.clear_loss()
        validation_metric_tracker.clear_metrics()

        train_writer.flush()
        validation_writer.flush()

        state = {'epoch': epoch,
                 'state_dict':generator.state_dict(),
                 'optimizer':opt_G.state_dict()}

        step = step + 1

        end = time.time()        
        
        train_loss = np.sum([np.mean(value) for key,value in train_loss_dict.items()])
        validation_loss = np.sum([np.mean(value) for key,value in validation_loss_dict.items()])
        
        sch_D.step(validation_loss)
        sch_G.step(validation_loss)
        
        tr_l1_loss = np.mean(train_loss_dict['l1'])
        tr_pl_loss = np.mean(train_loss_dict['pl'])
        all_loss.append([train_loss, validation_loss, tr_l1_loss, tr_pl_loss])   
        
        #print(f'loss is {loss}, per loss: {ploss}, l1loss: {l1loss}')
        
        print('Epoch: %.d' %epoch, end='   ')  
        print('Time: %.4f' %(end-start))
        print('Train Loss: %.4f' %train_loss)
        print('Validation Loss: %.4f' %validation_loss)
        print('Tr l1 loss: %.4f' %tr_l1_loss)
        print('Tr pl loss: %.4f' %tr_pl_loss)
        print('')
        
        
        #saves best network according to minimal validation loss
        if currentLoss == 0 or validation_loss < currentLoss:
            currentLoss = validation_loss

            try:
                os.system('rm ' + currentSave)
            except:
                pass
            
            currentSave = (directory + 'residual_booster_network_epoch_%.d_val_%.4f.pth' %(epoch,currentLoss))
            torch.save(state,currentSave)

        # saves intermediate networks according to save interval
        if epoch % save_interval == 0:
            torch.save(state,directory+'checkpoint_epoch_%.d_val_%.4f.pth' %(epoch,validation_loss))
            print_loss(all_loss, plot_name)
    
    torch.save(state,directory+save_directory+'.pth')
    #torch.onnx.export(model,torch.zeros((3,2,144,144,32),device='cuda'),directory+save_directory+'.onnx',opset_version=11)
    print_loss(all_loss, plot_name)
