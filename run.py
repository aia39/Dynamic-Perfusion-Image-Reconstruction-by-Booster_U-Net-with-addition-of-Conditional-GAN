#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 01:15:54 2026

@author: maistiak
"""


import sys
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append('loss_functions')
sys.path.append('template_models')
sys.path.append('utils')

import os.path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from supportingFunctions import *
import argparse
import scipy.io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from booster_runet import *
from loadData import *
import time
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------------------------------
#
# visualize results of trained network
# run in command-line 
# command: python visualize test_mode nset gif
#                 -test_mode (integer): selects test dataset to visualize [default: 0]
#                 -nset (integer 0|1|2): selects slice group to visualize [default: 1]
#                 -gif (boolean True|False): choose whether to generate movie or not, 
#                                            saves to results directory [default: 1 (True)]
#
# ------------------------------------------------------------------------------------------------------



# Modify this line with location of trained network
directory = 'trainedNetwork/'


with open(directory+'parameters.pkl','rb') as f:
    parameters = pickle.load(f)

with open(directory+'scale_factors.pkl','rb') as f:
    mean,std = pickle.load(f)

state = torch.load(directory+'residual_booster_network_epoch_0_val_0.7009.pth', weights_only=True)

parser = argparse.ArgumentParser(description = 'visualize results of network')
parser.add_argument('--test_mode', type = int, default=1)
parser.add_argument('--nset',type = int, default=1)
parser.add_argument('--gif',type = int, default=1)
args = parser.parse_args()

savelocation = 'results/testset' + str(args.test_mode) + '/set' + str(args.nset) + '/'

if not os.path.exists(savelocation):
    os.makedirs(savelocation)

testset = loadData(N=parameters['N'], split='test', test_mode=args.test_mode, 
                   directory=directory)

test_loader = torch.utils.data.DataLoader(testset,batch_size=parameters['batch'])

model = booster_runet(in_features=parameters['in_features'], out_features=parameters['out_features'], resolution=parameters['resolution'], layers=parameters['layers'], dp=parameters['dp']).to(device)

model.load_state_dict(state['state_dict'])

inputs,outputs,targets,total_time = test_network(model,test_loader,device)

inputs = inputs*std + mean
outputs = outputs*std + mean
targets = targets*std + mean


test_metric_tracker = metric_tracker(mean,std)

## cropped to heart fov
quarter_fov = outputs.shape[2]//4
half_fov = outputs.shape[2]//2

outputs_crp = outputs[:, :, quarter_fov:quarter_fov+half_fov+1, quarter_fov:quarter_fov+half_fov+1, :]
targets_crp = targets[:, :, quarter_fov:quarter_fov+half_fov+1, quarter_fov:quarter_fov+half_fov+1, :]


### Taking only abs before blur metric ###
outputs_crp = np.abs(outputs_crp[:,0,...]+1j*outputs_crp[:,1,...])
targets_crp = np.abs(targets_crp[:,0,...]+1j*targets_crp[:,1,...])
outputs = np.abs(outputs[:,0,...]+1j*outputs[:,1,...])
targets = np.abs(targets[:,0,...]+1j*targets[:,1,...])

## Calculate evaluation metrics
test_metric_tracker.calculate_metrics(outputs_crp,targets_crp,magnitude=True)              #(outputs,targets)

test_metric_dict = test_metric_tracker.return_metrics()

for key,value in test_metric_dict.items():
    test_metric_dict[key] = {'mean': np.mean(value), 'std': np.std(value)}


nb,sy,sx,nt = targets_crp.shape
outputs_blur = np.reshape(outputs_crp,(nb*nt,sy,sx))
targets_blur = np.reshape(targets_crp,(nb*nt,sy,sx))

total_o_blur = 0
total_t_blur = 0
for it in range(outputs_blur.shape[0]):
    blur_o = blur_metric(outputs_blur[it,:,:])
    blur_t = blur_metric(targets_blur[it,:,:])
    total_o_blur += blur_o
    total_t_blur += blur_t
print(f'Output quarter blur: {total_o_blur/it}')
print(f'Target quarter blur: {total_t_blur/it}')

print('')
print('TIME: %.4f'  %total_time)
print('Quarter SSIM: %.4f +/- %.4f' %(test_metric_dict['ssim']['mean'],test_metric_dict['ssim']['std']))
print('Quarter PSNR: %.4f +/- %.4f' %(test_metric_dict['psnr']['mean'],test_metric_dict['psnr']['std']))
print('Quarter NRMSE: %.4f +/- %.4f' %(test_metric_dict['nrmse']['mean'],test_metric_dict['nrmse']['std']))


test_metric_tracker.clear_metrics()
test_metric_tracker = metric_tracker(mean,std)
test_metric_tracker.calculate_metrics(outputs,targets,magnitude=True)              #(outputs,targets)
test_metric_dict = test_metric_tracker.return_metrics()

for key,value in test_metric_dict.items():
    test_metric_dict[key] = {'mean': np.mean(value), 'std': np.std(value)}

nb,sy,sx,nt = targets.shape
outputs_blur = np.reshape(outputs,(nb*nt,sy,sx))
targets_blur = np.reshape(targets,(nb*nt,sy,sx))

total_o_blur = 0
total_t_blur = 0
for it in range(outputs_blur.shape[0]):
    blur_o = blur_metric(outputs_blur[it,:,:])
    blur_t = blur_metric(targets_blur[it,:,:])
    total_o_blur += blur_o
    total_t_blur += blur_t
    
print(f'Output blur: {total_o_blur/it}')
print(f'Target blur: {total_t_blur/it}')
print('')
print('TIME: %.4f'  %total_time)
print('SSIM: %.4f +/- %.4f' %(test_metric_dict['ssim']['mean'],test_metric_dict['ssim']['std']))
print('PSNR: %.4f +/- %.4f' %(test_metric_dict['psnr']['mean'],test_metric_dict['psnr']['std']))
print('NRMSE: %.4f +/- %.4f' %(test_metric_dict['nrmse']['mean'],test_metric_dict['nrmse']['std']))


nb,ch,sx,sy,nt = targets.shape

inputs = inputs*std + mean
outputs = outputs*std + mean
targets = targets*std + mean

inputs = np.abs(r2c(inputs,dim='first'))
outputs = np.abs(r2c(outputs,dim='first'))
targets = np.abs(r2c(targets,dim='first'))

inputs = np.transpose(inputs,(0,3,1,2))
outputs = np.transpose(outputs,(0,3,1,2))
targets = np.transpose(targets,(0,3,1,2))

inputs = np.reshape(inputs,(nb*nt,sx,sy))
outputs = np.reshape(outputs,(nb*nt,sx,sy))
targets = np.reshape(targets,(nb*nt,sx,sy))

nt,sx,sy = targets.shape
nsl = 3   #####

inputs = np.reshape(inputs,(nsl,nt//nsl,sx,sy))
outputs = np.reshape(outputs,(nsl,nt//nsl,sx,sy))
targets = np.reshape(targets,(nsl,nt//nsl,sx,sy))

nsl,nfr,sx,sy = targets.shape
clip = (nfr//3)*args.nset


inputs = normalize(inputs)
outputs = normalize(outputs)
targets = normalize(targets)


inputs = brighten(inputs,0.4)
outputs = brighten(outputs,0.4)
targets = brighten(targets,0.4)


diff = np.abs(targets - outputs)

nsl,nt,sy,sx = targets.shape


for i in range(inputs.shape[1]):
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))  # 2 rows, 1 column
    axes[0,0].imshow(inputs[0,i,:,:], cmap = 'gray')
    axes[0,1].imshow(outputs[0,i,:,:], cmap = 'gray')
    axes[0,2].imshow(targets[0,i,:,:], cmap = 'gray')
    axes[1,0].imshow(inputs[1,i,:,:], cmap = 'gray')
    axes[1,1].imshow(outputs[1,i,:,:], cmap = 'gray')
    axes[1,2].imshow(targets[1,i,:,:], cmap = 'gray')
    axes[2,0].imshow(inputs[2,i,:,:], cmap = 'gray')
    axes[2,1].imshow(outputs[2,i,:,:], cmap = 'gray')
    axes[2,2].imshow(targets[2,i,:,:], cmap = 'gray')
    
    fig.suptitle(f"Frame: {i}", fontsize=16)
    plt.tight_layout()
    plt.show()

print('generating movie...')
print('saving movie to' + savelocation)

'''
####### Generating individual Gif ##########
import imageio.v2 as imageio
recon_list = [outputs[0,j,:,:] for j in range(outputs.shape[1])]
input_list = [inputs[0,j,:,:] for j in range(inputs.shape[1])]
gt_list = [targets[0,j,:,:] for j in range(targets.shape[1])]

def generate_gif(normalized_image_list, name):
    denorm_image_list = [(img * 255).astype(np.uint8) for img in normalized_image_list]
    image_list_rgb = [np.stack([img]*3, axis=-1) for img in denorm_image_list]
    imageio.mimsave(name+'.gif', image_list_rgb, duration=0.1, loop = 0)

generate_gif(recon_list, 'ungated_multi_band_recon_3D_sl1')
generate_gif(input_list, 'ungated_multi_band_input_3D_sl1')
generate_gif(gt_list, 'ungated_multi_band_gt_3D_sl1')
'''