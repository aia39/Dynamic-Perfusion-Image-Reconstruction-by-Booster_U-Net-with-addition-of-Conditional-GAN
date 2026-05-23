import sys
sys.path.append('utils')

import os.path
import numpy as np
import glob
import random
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io
from  supportingFunctions import *
from torch.utils.data import Dataset

def half_FOV(image):
    h, w, _, _, _ = image.shape
    image = image[h//4:3*h//4, w//4:3*w//4, :, :, :]
    return image

def specific_FOV(img):
    # Get dimensions (expand to 5D safely like MATLAB)
    H, W, _, _, _ = img.shape

    offset = 10
    offset_add_row = 10
    offset_add_col = 12

    # Calculate half sizes
    h_half = H // 4
    w_half = W // 4

    # Center coordinates
    centerH = H // 2
    centerW = W // 2

    # Define crop bounds (equivalent to MATLAB 1-based indexing, adjusted for Python 0-based)
    row_start = centerH - h_half - offset_add_col - offset
    row_end   = centerH + h_half + offset_add_col - offset
    col_start = centerW - w_half - offset_add_col - offset
    col_end   = centerW + w_half + offset_add_col - offset

    # Clip bounds to image size (Python safe indexing)
    row_start = max(row_start, 0)
    row_end = min(row_end, H)
    col_start = max(col_start, 0)
    col_end = min(col_end, W)

    # Crop image
    cropped = img[row_start:row_end, col_start:col_end, ...]

    return cropped


def load_ungated_sms(N=32, split='train', test_mode=None, val_mode=None, half_fov = False, reduced_rays = False):

# -------------------------------------------------------------------------------------------
#
#     inputs,targets = load_sms_data(N=32, split='train', test_mode=None, val_mode=None)
#                      - loads and prepares SMS datasets for training and testing the 
#                        residual booster network
#                      - helper function for loadData()
#                      - SMS datasets have dimensions [sx,sy,nt,nsets,nsl]
#                            - sx: spatial dimension
#                            - sy: spaitla dimension
#                            - nt: number of time frames
#                            - nsets: number of slice groups
#                            - nsl: number of simultaneously exicted slices
# -------------------------------------------------------------------------------------------
#    
#     inputs (Ungated 2D radial SMS myocardial perfusion datasets)
#
#        -N (integer: 8|16|32): number of training time frames for each batch [default: 32]
#        -split (string: 'train|val|test'): load selected datasets [default:'train']
#        -test_mode (integer: 0|1|2|3|4|5): selects test set to load,
#                                           loads all test datasets if not specified [default: None]
#        -val_mode (integer: 0|1): selects validation set to load, 
#                                  loads all validation datasets if not specified [default: None]
#
# -------------------------------------------------------------------------------------------
#     outputs
#        - inputs [nb,ch,sx,sy,nt]: inputs to feed into the residual booster network 
#        - targets [nb,ch,sx,sy,nt]: PT-STCR references for training the residual booster network
#              -nb: batch dimension
#              -ch: real and imaginary components (2)
#              -sx: spatial dimension
#              -sy: spatial dimension 
#              -nt: number of time frames chosen by N 
#
#

    if split == 'test':

        datasets = glob.glob('/v/raid1b/backup/maistiak/ReconData/ReconData_binnednew/testing/*.mat')   #39 dataset


    if split == 'val':

        datasets = glob.glob('/v/raid1b/backup/maistiak/ReconData/ReconData_binnednew/valid/*.mat')   #39 dataset


            
    if split == 'train':

        datasets = glob.glob('/v/raid1b/backup/maistiak/ReconData/ReconData_binnednew/training/*.mat')   #39 dataset


    
    inputs = []
    targets = []
    for i in range(len(datasets)):
        print(i, datasets[i])

        f = scipy.io.loadmat(datasets[i],variable_names=['Image_init_sys','Image_init_dia','Image_sys','Image_dia'])
        
        
        sys_init = f['Image_init_sys']
        sys = f['Image_sys']
        dia_init = f['Image_init_dia']
        dia = f['Image_dia']
        
        # make all frames for sys init and sys equal as in less rays there won't be same frames
        if(reduced_rays == True):
           sys_nof = min(sys_init.shape[2], sys.shape[2])
           sys = sys[:,:,:sys_nof,:,:]
           sys_init = sys_init[:,:,:sys_nof,:,:]
           
           dia_nof = min(dia_init.shape[2], dia.shape[2])
           dia = dia[:,:,:dia_nof,:,:]
           dia_init = dia_init[:,:,:dia_nof,:,:]
           
           sx,sy,nfr_sys,nsets,nsl = sys.shape
           sx,sy,nfr_dia,nsets,nsl = dia.shape
        else:   
            sx,sy,nfr_sys,nsets,nsl = f['Image_sys'].shape   
            sx,sy,nfr_dia,nsets,nsl = f['Image_dia'].shape
            
        
        
        if(half_fov == True):
            ### Half FOV ### for pre-processing
            sys_init = half_FOV(sys_init)
            sys = half_FOV(sys)
            dia_init = half_FOV(dia_init)
            dia = half_FOV(dia)
        
        ###############################################
        ##### cropping to only myocardium area ########
        ###############################################

        #sys_init = specific_FOV(sys_init)
        #sys = specific_FOV(sys)
        #dia_init = specific_FOV(dia_init)
        #dia = specific_FOV(dia)
        #print(f'Modified sys size is {sys.shape}')
        sx,sy,nfr_sys,nsets,nsl = sys.shape   
        sx,sy,nfr_dia,nsets,nsl = dia.shape
        
        
        ll_sys = N*(nfr_sys//N)
        ll_dia = N*(nfr_dia//N)

        sys_init = sys_init[:,:,:ll_sys,:,:]
        sys = sys[:,:,:ll_sys,:,:]
        
        
        dia_init = dia_init[:,:,:ll_dia,:,:]
        dia = dia[:,:,:ll_dia,:,:]

        sys_init = np.transpose(sys_init,[0,1,4,3,2])
        sys = np.transpose(sys,[0,1,4,3,2])
        dia_init = np.transpose(dia_init,[0,1,4,3,2])
        dia = np.transpose(dia,[0,1,4,3,2])

        sx,sy,nsl,nsets,nfr_sys = sys.shape
        sys_init = np.reshape(sys_init,(sx,sy,nsl*nsets*nfr_sys))
        sys = np.reshape(sys,(sx,sy,nsl*nsets*nfr_sys))
        
        sx,sy,nsl,nsets,nfr_dia = dia.shape  ##
        dia_init = np.reshape(dia_init,(sx,sy,nsl*nsets*nfr_dia))
        dia = np.reshape(dia,(sx,sy,nsl*nsets*nfr_dia))

        sx,sy,nfr_sys = sys.shape

        sys_init = np.reshape(sys_init,(sx,sy,nfr_sys//N,N))
        sys = np.reshape(sys,(sx,sy,nfr_sys//N,N))
        
        sx,sy,nfr_dia = dia.shape   ##
        dia_init = np.reshape(dia_init,(sx,sy,nfr_dia//N,N))
        dia = np.reshape(dia,(sx,sy,nfr_dia//N,N))
        #print(sys.shape)
        inputs.append(sys_init)   #144,144,18,32 of 12 elements
        inputs.append(dia_init)
        targets.append(sys)
        targets.append(dia)
        print(sys_init.shape)
        
        
    inputs = np.concatenate(inputs,2)   #144,144,189,32
    targets = np.concatenate(targets,2)
    
    inputs = np.transpose(inputs,(2,0,1,3))   #189,144,144,32
    targets = np.transpose(targets,(2,0,1,3))
    
    inputs = c2r(inputs,dim='first')    #189,2,144,144,32
    targets = c2r(targets,dim='first')
    
    return inputs,targets



class loadData(Dataset):
    def __init__(self,N=32, split='train', test_mode=None, val_mode=None, do_transform=False, directory=None, reduced_rays = False):
        
# -------------------------------------------------------------------------------------------
#
#     inputs,targets = loadData(N=32, split='train', test_mode=None, val_mode=None)
#                      - loads and prepares SMS datasets for training and testing the 
#                        residual booster network
#                      - SMS datasets have dimensions [sx,sy,nt,nsets,nsl]
#                            - sx: spatial dimension
#                            - sy: spaitla dimension
#                            - nt: number of time frames
#                            - nsets: number of slice groups
#                            - nsl: number of simultaneously exicted slices
# -------------------------------------------------------------------------------------------
#    
#     inputs (2D radial SMS myocardial perfusion datasets)
#
#        -N (integer: 8|16|32): number of training time frames for each batch [default: 32]
#        -split (string: 'train|val|test'): load selected datasets [default:'train']
#        -test_mode (integer: 0,1,2,3,4,5): selects test set to load,
#                                           loads all test datasets if not specified [default: None]
#        -val_mode (integer: 0,1): selects validation set to load, 
#                                  loads all validation datasets if not specified [default: None]
#        -do_transform (boolean: True|False): performs data augmentations [default: True]
#        -directory (string): save location for network and other parameters [default: None]
# -------------------------------------------------------------------------------------------
#     outputs
#        - inputs [nb,ch,sx,sy,nt]: inputs to feed into the residual booster network 
#        - targets [nb,ch,sx,sy,nt]: PT-STCR references for training the residual booster network
#              -nb: batch dimension
#              -ch: real and imaginary components (2)
#              -sx: spatial dimension
#              -sy: spatial dimension 
#              -nt: number of time frames chosen by N 
#
#

        inputs,targets = load_ungated_sms(split=split, N=N, test_mode=test_mode, val_mode=val_mode, half_fov = False, reduced_rays=reduced_rays)

        self.do_transform = do_transform
        self.inputs = inputs
        self.targets = targets
        
        if split == 'train':
            if os.path.isfile(directory+'scale_factors.pkl'):

                with open(directory+'scale_factors.pkl','rb') as f:
                    self.mean,self.std = pickle.load(f)

            else:
                self.mean = np.mean(self.inputs)
                self.std = np.std(self.inputs)

                with open(directory+'scale_factors.pkl','wb') as f:
                    pickle.dump([self.mean,self.std],f)
                    
            print(f"Mean is {self.mean} and s.deviation is {self.std}")
            
            self.inputs = (self.inputs - self.mean)/self.std
            self.targets = (self.targets  - self.mean)/self.std

        else:
            if os.path.isfile(directory+'scale_factors.pkl'):
                with open(directory+'scale_factors.pkl','rb') as f:
                    self.mean,self.std = pickle.load(f)
            else:
                print('scale file does not exist. run script in training mode first.')
                sys.exit()

            self.inputs = (self.inputs - self.mean)/self.std
            self.targets = (self.targets - self.mean)/self.std

    def __getitem__(self,index):
        if self.do_transform:
            if random.uniform(0,1) < 1:
                inputs,targets = toPIL(self.inputs[index],self.targets[index])
                if random.uniform(0,1) < 0.5:
                    if random.uniform(0,1) < 0.5:
                        inputs,targets = horizontal_flip(inputs,targets)
                    if random.uniform(0,1) < 0.5:
                        inputs,targets = vertical_flip(inputs,targets)
                elif random.uniform(0,1) < 0.5:
                    inputs,targets = shear(inputs,targets)
                else:
                    if random.uniform(0,1) < 0.5:
                        inputs,targets = translate(inputs,targets)
                    if random.uniform(0,1) < 0.5:
                        inputs,targets = rotate(inputs,targets)
                inputs,targets = toTensor(inputs,targets)            
            else:
                inputs = self.inputs[index]
                targets = self.targets[index]
        else:
            inputs = self.inputs[index]
            targets = self.targets[index]

        return inputs,targets

    def __len__(self):
        return len(self.targets)
        
if __name__ == '__main__':
    loadData(split='test',directory='trainedNetwork/18Aug_0350am/',test_mode=1,do_transform=False)
