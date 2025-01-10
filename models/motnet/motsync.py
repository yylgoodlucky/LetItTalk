import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
from models.stylegan.psp.encoder import psp_encoders
from models.stylegan.stylesync.model import EqualLinear, PixelNorm, Normalize

class MotSync(nn.Module):
    def __init__(self, 
                 opts: dict,
                 n_mlp: int,
                 embeding_dim: int,
                 lr_mlp=0.01,
                 ):
        super(MotSync, self).__init__()
        self.opts = opts
        self.n_mlp = n_mlp
        self.embeding_dim = embeding_dim
        
        self.img_encoder = self.set_encoder()
        
        layers = [Normalize(),
                  EqualLinear(1024, self.embeding_dim, lr_mul=lr_mlp, activation=None),
                  ReLU(self.embeding_dim)
                  ]
        for i in range(self.n_mlp):
            layers.append(EqualLinear(self.embeding_dim, self.embeding_dim, lr_mul=lr_mlp, activation=None))
            layers.append(ReLU(self.embeding_dim))
        self.mot_encoder = nn.Sequential(*layers)
        
        
    def set_encoder(self):
        """ setting for image encoder.
        """
        if self.opts['encoder_type'] == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts['encoder_type'] == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts['encoder_type'] == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts['encoder_type']))
        return encoder

    def forward(self, 
                img: torch.Tensor, 
                mot: torch.Tensor):
        """
        args:
            img: (batch_size, 3, img_size, img_size)
            mot: (batch_size, 1024)
        return:
            similarity score: (batch_size, 1)
        """
        img_embedding = self.img_encoder(img)
        mot_embedding = self.mot_encoder(mot)
        mot_embedding = mot_embedding.view(mot_embedding.size(0), -1)
        img_embedding = img_embedding.view(img_embedding.size(0), -1)

        img_embedding = F.normalize(img_embedding, p=2, dim=1)
        mot_embedding = F.normalize(mot_embedding, p=2, dim=1)
        return img_embedding, mot_embedding
    
    
if __name__ == '__main__':
    model = MotSync(opts=None, n_mlp=3, embeding_dim=512)
    img_batch = torch.ones(1, 3, 512, 512)
    mot_batch = torch.ones(1, 1024)
    x = model(img_batch, mot_batch)
    print(x[0].shape, x[1].shape)