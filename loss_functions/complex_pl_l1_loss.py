import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary
from transformers import AutoModel

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

from scipy.ndimage import zoom

def visualize_features(feats, img_size=(144,144), method="pca", dino = 2):
    B, N, D = feats.shape
    H, W = img_size

    if(dino == 3):
        # drop CLS and register tokens
        feats = feats[:, 5:, :]   # [B, N-5, D]
        patch_size = 16
    else:
        feats = feats[:, 1:, :]   # [B, N-1, D]
        patch_size = 14
    # reshape to patch grid
    h, w = H // patch_size, W // patch_size
    ##feats = feats.reshape(B, h, w, D)   # [B, h, w, D]
    #feats = feats.permute(0, 3, 1, 2)   # [B, D, h, w]

    # upsample to full resolution
    ##feats_up = F.interpolate(feats, size=(H, W), mode="bilinear", align_corners=False)  # [B, D, H, W]
    
    # reduce D to 3 for visualization
    if method == "norm":
        vis = feats.norm(dim=1)  # [B, H, W]
    elif method == "pca":
        # flatten spatial → apply PCA
        flat = feats.reshape(-1, D).cpu().numpy()  # [H*W, D]   feats_up[30].permute(1,2,0).reshape(-1, D).cpu().numpy()
        pca = PCA(n_components=3)
        proj = pca.fit_transform(flat)  # [H*W, 3]
        proj = (proj - proj.min(0)) / (proj.max(0)-proj.min(0))
        nth = 30
        vis = proj[nth*h*w : (nth+1)*h*w, 0].reshape(h, w)    #proj.reshape(H, W, 3)
        #feats_up = F.interpolate(vis, size=(H, W), mode="bilinear", align_corners=False)  # [B, D, H, W]
        
        vis = zoom(vis, zoom=H/h, order=1)
        
    else:
        raise ValueError("method must be 'norm' or 'pca'")

    return vis


class DINOAutoencoder(nn.Module):
    def __init__(self, encoder, latent_dim=256):
        super().__init__()
        self.encoder = encoder

        # Get embedding dimension from DINOv2 backbone
        self.enc_dim = encoder.config.hidden_size
        
    def forward(self, x):
            visualize = 0    #######Change to 1 to visualize feature   
            # DINOv2 expects (B, C, H, W)
            
            #mean, std = 0.5, 0.2
            #x = (x - mean) / std
                 
            with torch.no_grad():
                features = self.encoder(x).last_hidden_state  # (B, N, D)
            #print(f'Feature shape is {features.shape}')
            
            
            if(visualize):
                vis = visualize_features(features, img_size=(144,144), method="pca", dino = 3)
                
                plt.subplot(1, 2, 1)
                plt.imshow(vis)
                plt.axis("off")
                plt.title("DINOv2 Feat")
                plt.colorbar()
                plt.subplot(1, 2, 2)
                plt.imshow(x[30,0,:,:].detach().cpu().numpy(), cmap='gray')
                plt.colorbar()
                plt.show()
            
            
            # Take CLS token or mean pool
            ##z = features.mean(dim=1)  # (B, D)            
            return features ##z


def visualize_PCA_vgg(feats, img):
    B, C, H, W = feats.shape
    feat_map = feats[30].permute(1,2,0).reshape(-1, C).detach().cpu().numpy()  # [H*W, C]
    
    # PCA on channel dimension
    pca = PCA(n_components=1)   # take top 3 PCs
    pcs = pca.fit_transform(feat_map)  # [H*W, 3]
    pcs_img = pcs.reshape(H, W, 1)
    
    # Normalize for visualization
    pcs_img = (pcs_img - pcs_img.min()) / (pcs_img.max() - pcs_img.min())
    
    # Resize back to input size
    pcs_img_up = F.interpolate(torch.tensor(pcs_img).permute(2,0,1).unsqueeze(0),
                               size=(144,144), mode="bilinear", align_corners=False)
    pcs_img_up = pcs_img_up.squeeze().squeeze().numpy()
    
    # Show
    plt.subplot(1,2,2)
    plt.imshow(img[30,0,:,:].detach().cpu().numpy(), cmap='gray')
    plt.title("Original Image")
    
    plt.subplot(1,2,1)
    plt.imshow(pcs_img_up, cmap ='gray')
    plt.title("PCA Feature")
    plt.colorbar()
    plt.show()
    

def visualize_PCA(feature, npca=1, frame = 30):
    patch_feats = feature[:, 1:, :].squeeze(0).detach().cpu().numpy()   #excluding CLS token
    nt, nfeat, hid_dem = patch_feats.shape
    #a = np.reshape(patch_feats,(2,-1,nfeat,hid_dem))
    a = patch_feats[frame,:,:]
    #r = 1  #real = 0, imaginary = 1
    #magnitude = np.sqrt(b[0,:,:]**2 + b[1,:,:]**2)
    #c = b[r,:,:]
    pca = PCA(n_components=npca)
    proj = pca.fit_transform(a)
    
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0))
    proj = proj * 255
    # Reshape back to patch grid
    num_patches = proj.shape[0]
    grid_size = int(np.sqrt(num_patches))
    proj_img = proj.reshape(grid_size, grid_size, npca).astype(np.uint8)
    
    plt.imshow(proj_img)
    plt.title("Spatial PCA on DINOv2 patch embeddings")
    plt.axis("off")
    plt.colorbar()
    plt.show()

    
def visualize_diff(sub, frame = 30):
    patch_feats = sub[:, 1:, :].squeeze(0).detach().cpu().numpy()   #excluding CLS token
    
    #nt, nfeat, hid_dem = patch_feats.shape
    #a = np.reshape(patch_feats,(2,-1,nfeat,hid_dem))
    proj = patch_feats[frame,:,2]
    #r = 1  #real = 0, imaginary = 1
    #magnitude = np.sqrt(b[0,:,:]**2 + b[1,:,:]**2)
    #c = b[r,:,:]
    #pca = PCA(n_components=npca)
    #proj = pca.fit_transform(a)
    
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0))
    proj = proj * 255
    # Reshape back to patch grid
    num_patches = proj.shape[0]
    grid_size = int(np.sqrt(num_patches))
    proj_img = proj.reshape(grid_size, grid_size).astype(np.uint8)
    
    plt.imshow(proj_img)
    plt.title("Subtraction on DINOv2 patches")
    plt.axis("off")
    plt.colorbar()
    plt.show()
    

class DINOv3encoder(nn.Module):
    def __init__(self, encoder, latent_dim=768):
        super().__init__()
        self.encoder = encoder

        # Get embedding dimension from DINOv2 backbone
        self.enc_dim = encoder.config.hidden_size
        
    def forward(self, x):
        #my_batch = torch.rand(256, 3, 144, 144)  # 480×640 instead of 224×224
        
        ## Normalize only (no resize)
        #mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
        #std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
        #my_batch = (x - mean) / std
        my_batch = x
        
        
        with torch.inference_mode():
            outputs = self.encoder(pixel_values=my_batch.to(self.encoder.device))
            
        
        #print("Last hidden state:", outputs.last_hidden_state.shape)  # [B, N_tokens, hidden_dim]
        #print("Pooled CLS token :", outputs.pooler_output.shape)
        sx = x.shape[-2]
        visualize = 0
        if(visualize):
            vis = visualize_features(outputs.last_hidden_state, img_size=(sx,sx), method="pca", dino = 3)
            
            plt.subplot(1, 2, 1)
            plt.imshow(vis)
            plt.axis("off")
            plt.title("DINOv3 Feat")
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(x[30,0,:,:].detach().cpu().numpy(), cmap='gray')
            plt.colorbar()
            plt.show()    
        return outputs.last_hidden_state   #[:, 5:, :]
        

def gaussian_weight_map(H, W, sigma=30):
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    cy, cx = H//2, W//2
    dist2 = (x-cx)**2 + (y-cy)**2
    g = torch.exp(-dist2/(2*sigma**2))
    return g / g.max()  # normalize to [0,1]


def CharbonnierLoss(pred, target, eps=1e-6):
    diff = pred - target
    loss = torch.mean(torch.sqrt(diff * diff + eps ** 2))
    return loss


def interpolating_to_highres(x):
    nb, nc, nx, ny = x.shape 
    #x_reshaped = x.view(-1, nc, nx, ny)  # (12*32, 3, 96, 96)
    
    des = 512
    # Interpolate spatially
    x_up = F.interpolate(x, size=(des, des), mode='bilinear', align_corners=False)
    
    # Restore original grouping
    #x_up = x_up.view(nb, nt, nc, des, des)
    
    return x_up


class complex_pl_l1_loss(nn.Module):
    def __init__(self,alpha,beta):
        super(complex_pl_l1_loss,self).__init__()

        self.alpha = alpha
        self.beta = beta
        #self.gamma = 0.5

        self.tracker = {'pl':[],'l1':[]}

    def perceptual_loss(self,predicted,target):
        
        #### DINOv2 inclusion ####
        #dinov2 = AutoModel.from_pretrained("facebook/dinov2-small")   #facebook/dinov2-large  dinov2-giant
        
        '''
        dinov2 = AutoModel.from_pretrained(
            "facebook/dinov2-small",
            cache_dir="/working/HF/hf_cache"
        )
        
        model = DINOAutoencoder(dinov2).cuda()
        '''
        
        ####### DINOv3 #######
        dinov3 = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m", device_map="auto")
        model = DINOv3encoder(dinov3)
        model.eval()
        
        nb,ch,sy,sx,nt = predicted.shape
        
        #print(f'Pred shape is {predicted.shape}')
        
        predicted = predicted.view(nb,ch,sy,sx,nt)
        
        #model = models.vgg16(pretrained=True).features[:16].cuda()     #in baseline boosterunet
        
        for param in model.parameters():
            param.requires_grad = False

        loss = 0
        
        
        '''
        ## Magnitude ##
        complex_img = predicted[:,0,:,:,:] + 1j * predicted[:,1,:,:,:]
        magnitude = torch.abs(complex_img)
        magnitude_pred = torch.unsqueeze(magnitude, axis=1)
        
        complex_img = target[:,0,:,:,:] + 1j * target[:,1,:,:,:]
        magnitude = torch.abs(complex_img)
        magnitude_targ = torch.unsqueeze(magnitude, axis=1)        
        
        ####
        predicted_rgb = magnitude_pred[:,0,None,:,:,:].repeat(1,3,1,1,1)
        target_rgb = magnitude_targ[:,0,None,:,:,:].repeat(1,3,1,1,1)
        ####
        '''
        #predicted = interpolating_to_highres(predicted)
        #target = interpolating_to_highres(target)
        
        #predicted_rgb = F.interpolate(predicted, size=(192, 192), mode='trilinear', align_corners=False)
        #target_rgb = F.interpolate(target, size=(192, 192), mode='trilinear', align_corners=False)
        #print(predicted.shape)

        for i in range(ch):
            predicted_rgb = predicted[:,i,None,:,:,:].repeat(1,3,1,1,1)
            target_rgb = target[:,i,None,:,:,:].repeat(1,3,1,1,1)
            
            predicted_rgb = predicted_rgb.permute((0,4,1,2,3)).contiguous()
            target_rgb = target_rgb.permute((0,4,1,2,3)).contiguous()

            nb,nt,ch,sy,sx = predicted_rgb.shape
            
            predicted_rgb = predicted_rgb.view((nb*nt,ch,sy,sx))
            target_rgb = target_rgb.view((nb*nt,ch,sy,sx))
            
            '''
            pred = model(predicted_rgb)
            vecA = pred.flatten(start_dim=1) if pred.ndim > 1 else pred.unsqueeze(0)
            gt = model(target_rgb)
            vecB = gt.flatten(start_dim=1) if gt.ndim > 1 else gt.unsqueeze(0)
            sim = F.cosine_similarity(vecA, vecB, dim=1, eps=1e-8)
            #sim = cosine_similarity(vecA.cpu().numpy(), vecB.cpu().numpy())[0,0]
            loss = loss + sim
            '''
            
            ##sub = model(target_rgb) - model(predicted_rgb)
            #visualize_diff(sub, frame = 30)
            ##loss = loss + torch.mean(sub)
            
            #visualize_PCA_vgg(pred, predicted_rgb)
            
            #predicted_rgb = interpolating_to_highres(predicted_rgb)
            #target_rgb = interpolating_to_highres(target_rgb)

            #loss = loss + torch.mean(((model(target_rgb) - model(predicted_rgb))**2))
            
            '''
            feat_indices = [5,8,12,15,20,25,40,51,53,57,59,79,81,82,91,96,104,107,109,114,126,135,152,163,165,173,179,181,187,189,198,214,227,237,245,292,296,329,330,331,345,364,381]
            #l2 = (model(target_rgb)[feat_indices] - model(predicted_rgb)[feat_indices])**2
            #loss = loss + torch.mean(l2)
            gt_feat = model(target_rgb)
            gt_feat = gt_feat[...,feat_indices]
            pred_feat = model(predicted_rgb)
            pred_feat = pred_feat[...,feat_indices]
            #print(gt_feat.shape, pred_feat.shape)
            l2 = (gt_feat - pred_feat)**2
            
            loss = loss + torch.mean(l2) 
            '''
            loss = loss + torch.mean(torch.abs((model(target_rgb) - model(predicted_rgb))))   #L1 on perceptual    
            #loss = loss + CharbonnierLoss(model(target_rgb),model(predicted_rgb))
        
        
        '''
        ## Weighted loss
        diff = (model(target_rgb) - model(predicted_rgb))
        patch_feats = diff[:, 5:, :]   #excluding CLS token
        nt, nfeat, hid_dem = patch_feats.shape
        grid_size = int(np.sqrt(nfeat))
        proj_img = patch_feats.reshape(nt, grid_size, grid_size, hid_dem)
        
        ### Multiplying by gaussian kernel
        ker = gaussian_weight_map(grid_size, grid_size, sigma = 3)
        #ker = torch.tensor(ker, device=diff.device, dtype=diff.dtype)
        ker = ker.clone().detach().requires_grad_(True)
        ker = torch.unsqueeze(ker, dim=0)
        ker = torch.unsqueeze(ker, dim=-1).cuda()
        
        weighted_diff = torch.mean((proj_img * ker)**2)
        loss = loss + weighted_diff 
        '''
            
        ###d = (1 / torch.sqrt(torch.tensor(2.0, device=pred.device))) * torch.sqrt(1 - loss/ch)
        return loss/ch      ##d.mean() 
    
    def l1_loss(self,predicted,target):
        
        nb,ch,sy,sx,nt = predicted.shape

        loss = 0
        
        for i in range(ch):    
            predicted_rgb = predicted[:,i,None,:,:,:].repeat(1,3,1,1,1)
            target_rgb = target[:,i,None,:,:,:].repeat(1,3,1,1,1)
            
            '''
            ## Magnitude based ##
            complex_img = predicted[:,0,:,:,:] + 1j * predicted[:,1,:,:,:]
            magnitude = torch.abs(complex_img)
            magnitude_pred = torch.unsqueeze(magnitude, axis=1)
            
            complex_img = target[:,0,:,:,:] + 1j * target[:,1,:,:,:]
            magnitude = torch.abs(complex_img)
            magnitude_targ = torch.unsqueeze(magnitude, axis=1)        
            
            ####
            predicted_rgb = magnitude_pred[:,0,None,:,:,:].repeat(1,3,1,1,1)
            target_rgb = magnitude_targ[:,0,None,:,:,:].repeat(1,3,1,1,1)
            ####
            '''
            
            predicted_rgb = predicted_rgb.permute((0,4,1,2,3)).contiguous()
            target_rgb = target_rgb.permute((0,4,1,2,3)).contiguous()
            
            nb,nt,ch,sy,sx = predicted_rgb.shape
            
            predicted_rgb = predicted_rgb.view((nb*nt,ch,sy,sx))
            target_rgb = target_rgb.view((nb*nt,ch,sy,sx))
            
            #loss = loss + CharbonnierLoss(predicted_rgb, target_rgb)
            
            '''
            ## Weighted L1 loss
            grid_size = sy
            diff = torch.abs(target_rgb - predicted_rgb)
            ker = gaussian_weight_map(grid_size, grid_size, sigma = 22)
            #ker = ker.clone().detach().requires_grad_(True)
            ker = torch.unsqueeze(ker, dim=0)
            ker = torch.unsqueeze(ker, dim=0).cuda()
            
            weighted_diff = torch.mean((diff * ker)**2)
            
            loss = loss + weighted_diff
            
            '''
            loss = loss + torch.mean(torch.abs(target_rgb - predicted_rgb))
            
        return loss/ch
    '''
    def gradient_loss(self, pred, target):
        loss = 0
        nb,ch,sy,sx,nt = pred.shape
        
        for i in range(ch):
            dy_pred = pred[:, i, 1:, :, :] - pred[:, i, :-1, :, :]
            dx_pred = pred[:, i, :, 1:, :] - pred[:, i, :, :-1, :]
        
            dy_target = target[:, i, 1:, :] - target[:, i, :-1, :]
            dx_target = target[:, i, :, 1:, :] - target[:, i, :, :-1, :]
            
            loss = loss + torch.mean(torch.abs(dy_pred - dy_target)) + torch.mean(torch.abs(dx_pred - dx_target))

        return loss/ch    
    
    '''
    import torch.fft


    def high_frequency_emphasis_loss(self, pred, target, alpha=1.0, temporal=False):
        """
        High-frequency emphasis loss for 5D data (B, C, H, W, T)
        Args:
            pred, target: tensors of shape (B, C, H, W, T)
            alpha: controls how strongly to emphasize high frequencies (1–3 typical)
            temporal: if True, also applies FFT along time dimension (3D FFT)
        """
        B, C, H, W, T = pred.shape
        
        pred = torch.complex(pred[:,0], pred[:,1])
        target = torch.complex(target[:,0], target[:,1])
        
        if temporal:
            # ----- Spatio-temporal FFT (3D FFT) -----
            pred_fft = torch.fft.fftshift(torch.fft.fftn(pred, dim=(2,3,4), norm='ortho'))
            target_fft = torch.fft.fftshift(torch.fft.fftn(target, dim=(2,3,4), norm='ortho'))
    
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)
    
            # Frequency weighting (radial)
            yy, xx, tt = torch.meshgrid(
                torch.linspace(-1, 1, H, device=pred.device),
                torch.linspace(-1, 1, W, device=pred.device),
                torch.linspace(-1, 1, T, device=pred.device),
                indexing='ij'
            )
            freq_radius = torch.sqrt(xx**2 + yy**2 + tt**2)
        else:
            # ----- Spatial-only FFT (2D FFT per frame) -----
            pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, dim=(2,3), norm='ortho'))
            target_fft = torch.fft.fftshift(torch.fft.fft2(target, dim=(2,3), norm='ortho'))
    
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)
    
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=pred.device),
                torch.linspace(-1, 1, W, device=pred.device),
                indexing='ij'
            )
            freq_radius = torch.sqrt(xx**2 + yy**2)
        
        # Emphasize high frequencies
        freq_weight = (freq_radius ** alpha)
        
        # Expand to broadcast over (B, C)
        freq_weight = freq_weight.view(1, 1, 144, 144, 1)
        loss = torch.mean(freq_weight * torch.abs(pred_mag - target_mag))
        return loss


    
    def return_loss(self):
        return self.tracker.copy()

    def clear_loss(self):
        for key,value in self.tracker.items():
            self.tracker[key] = []

    def forward(self,outputs,targets, verbose = False):
        pl = self.alpha*self.perceptual_loss(outputs,targets)
        l1 = self.beta*self.l1_loss(outputs,targets)
        #hf = 2*self.high_frequency_emphasis_loss(outputs, targets, alpha=1.8)
        #gl = self.gamma*self.gradient_loss(outputs,targets)
        
        #print(f'Loss values: {self.perceptual_loss(outputs,targets)}, {self.l1_loss(outputs,targets)}')        
        #print(f'Gradient loss: {gl.item()}')
        self.tracker['pl'].append(pl.detach().cpu().numpy())
        self.tracker['l1'].append(l1.detach().cpu().numpy())
        
        loss = pl + l1 #+ gl
        
        return loss, pl, l1
