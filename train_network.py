#############
##current exp: DINOv3 perceptual + L1 loss 30 rays with alpha = 3 with adversarial training
#############

import time
import pickle
import os.path
import scipy.io
from residual_booster_network_cgan import *
import argparse



if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-fr", "--frames", type=int, default=32, help='Number of fixed frames')
    ap.add_argument("-bs", "--batch", type=int, default= 5, help='Batch size')
    ap.add_argument("-in", "--in_ch", type=int, default= 2, help='input channels, 2 for real and imaginary')
    ap.add_argument("-out", "--out_ch", type=int, default= 2, help='output channels, 2 for real and imaginary')
    ap.add_argument("-lay", "--layers", type=int, default= 3, help='U-Net layers')
    ap.add_argument("-dp", "--dp", type=str, default='True', help='Dropout enabled or not')
    ap.add_argument("-re", "--resolution", type=int, default= 64, help='Resolution for 3D kernel (time dimension of kernel)')
    ap.add_argument("-alp", "--alpha", type=float, default= 3.0, help='Alpha co-efficient for perceptual loss')
    ap.add_argument("-bet", "--beta", type=float, default= 1.0, help='Beta co-efficient for L1 loss')
    ap.add_argument("-lr", "--learningr", type=float, default=0.0003, help='Learning rate')
    ap.add_argument("-epoch", "--nepoch", type=int, default=100, help='Number of epochs')
    ap.add_argument("-s_int", "--save_interv", type=int, default=20, help='After how many epoch it will save weight')
    ap.add_argument("-ngpu", "--gpun", type=str, default='0', help='Which GPU in ebe will it use in case of multiple GPUs')
    ap.add_argument("-plt", "--plt_name", type=str, default='case97_allloss.png', help='Name of the saved loss curve graph')
    
    
    args = vars(ap.parse_args())  
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args["gpun"]   ##ebe number 2 gpu will work as cuda:0 here
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    plot_name = args["plt_name"]
    #number of frames to train on
    N = args["frames"]  ##32
    #batch size
    batch = args["batch"]   #3
    #number of input features
    in_features = args["in_ch"]  #2
    #number of output features
    out_features = args["out_ch"]  #2
    #filter width
    resolution = args["resolution"] #64
    #number of layers
    layers = args["layers"]
    #drop out
    dp = args["dp"]
    #perceptual loss weight
    alpha = args["alpha"] ##1.5   #0.04
    #mean absolute error weight
    beta = args["beta"]
    #learning rate
    lr = args["learningr"]   #0.0003
    #number of training epochs
    nepoch = args["nepoch"]  #100
    #interval at which to save checkpoint networks
    save_interval = args["save_interv"]
    #save directory
    directory = 'trainedNetwork_regGT/'+datetime.now().strftime("%d%b_%I%M%P") + '/'
    #name of final network to save
    save_directory = 'residual_booster_network'
    #pre-train network with gated variant
    #pretrain = 'v/raid1b/backup/maistiak/deep_image_prior/deep-image-prior/sota/Booster_Unet/Deep-learning-reconstruction-Radial-SMS-perfusion/ungated/trainedNetwork/13Jul_0937am-20250418T172441Z-001/13Jul_0937am/residual_booster_network.pth'
    pretrain = None
    if not os.path.exists(directory):
        os.makedirs(directory)

    parameters = {'N':N, 'batch':batch, 'in_features':in_features,'out_features':out_features,'resolution':resolution, 'layers':layers,'dp':dp,'save_directory':save_directory}

    with open(directory+'parameters.pkl','wb') as f:
        pickle.dump(parameters,f)

    residual_booster_network(N,batch,in_features,out_features,resolution,layers,dp,alpha,beta,lr,nepoch,save_interval,directory,save_directory,pretrain,device,plot_name)
