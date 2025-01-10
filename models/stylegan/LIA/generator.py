from torch import nn
from .encoder import Encoder
from .styledecoder import Synthesis


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(size, style_dim, motion_dim, blur_kernel, channel_multiplier)

    def get_direction(self):
        return self.dec.direction(None)

    def synthesis(self, wa, alpha, feat):
        img = self.dec(wa, alpha, feat)

        return img

    def forward(self, img_source, img_drive, h_start=None):
        wa, alpha, feats, motion = self.enc(img_source, img_drive, h_start)
        img_recon = self.dec(wa, alpha, feats, motion)

        return img_recon


if __name__ == "__main__":
    import torch
    generator = Generator(size=512, style_dim=512, motion_dim=20, channel_multiplier=1).cuda()
    
    src = torch.ones(1, 3, 512, 512).cuda()
    ref = torch.ones(1, 3, 512, 512).cuda()
    mot = torch.ones(1, 1024).cuda()
    
    gen = generator(src, ref, mot)
    
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    
    print(get_parameter_number(generator))
    print(gen.shape)