import torch
import lpips
import torch.nn as nn
import torch.nn.functional as F



class L1_loss(torch.nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
        
    def forward(self, src_img: torch.Tensor, tgt_img: torch.Tensor, reduction='none'):
        loss = F.l1_loss(src_img, tgt_img, reduction=reduction)
        return loss
    

class LPIPSLoss(nn.Module):
    def __init__(self, 
            use_input_norm=True,
            range_norm=False,):
        super(LPIPSLoss, self).__init__()
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False).eval()
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        if self.range_norm:
            pred   = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred   = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return lpips_loss.mean()


class Cosine_loss(nn.Module):
    def __init__(self):
        super(Cosine_loss, self).__init__()
        self.logloss = nn.BCELoss()  # BEC的输入变量的取值范围[0,1]
        
    def forward(self, f_imbed, m_imbed, label):
        d = nn.functional.cosine_similarity(f_imbed, m_imbed)
        loss = self.logloss(d.unsqueeze(1), label)

        return loss
    

class GradientVariance(torch.nn.Module):
    """Class for calculating GV loss between to RGB images
       :parameter
       patch_size : int, scalar, size of the patches extracted from the gt and predicted images
       cpu : bool,  whether to run calculation on cpu or gpu
        """
    def __init__(self, patch_size):
        super(GradientVariance, self).__init__()
        self.patch_size = patch_size
        # Sobel kernel for the gradient map calculation
        self.register_buffer('kernel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3))
        self.register_buffer('kernel_y', torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).reshape(1, 1, 3, 3))

        # operation for unfolding image into non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)

    def forward(self, output, target, reduction='mean'):
        # converting RGB image to grayscale
        gray_output = TF.rgb_to_grayscale(output)
        gray_target = TF.rgb_to_grayscale(target)

        # calculation of the gradient maps of x and y directions
        gx_target = F.conv2d(gray_target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(gray_target, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(gray_output, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(gray_output, self.kernel_y, stride=1, padding=1)

        # unfolding image to patches
        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        # calculation of variance of each patch
        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        # loss function as a MSE between variances of patches extracted from gradient maps
        gradvar_loss = F.mse_loss(var_target_x, var_output_x, reduction='none') + F.mse_loss(var_target_y, var_output_y, reduction='none')
        gradvar_loss = gradvar_loss.mean(dim=1)
        if reduction == 'mean':
            gradvar_loss = gradvar_loss.mean()

        return gradvar_loss