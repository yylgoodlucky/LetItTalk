import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from os.path import join

from models.vae.vq_vae import VQVAE
from utils.loss_util import PerceptualLoss
from data.base_dataset import BaseDataset
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self, 
                 opt, 
                 mode, 
                 device='cuda'):
        self.opt = opt
        self.mode = mode
        self.device = device
    
    
    def setup_logger(self):
        """
        Setup logger for training
        """
        config = self.opt['logger']
        exp_dir = join(config['dir'], config['name'])
        os.makedirs(exp_dir, exist_ok=True)
        
        vis_dir = join(exp_dir, 'vis')
        log_dir = join(exp_dir, 'log')
        ckpt_dir = join(exp_dir, 'ckpt')
        
        for dir in [vis_dir, log_dir, ckpt_dir]:
            os.makedirs(dir, exist_ok=True)
        
        self.vis_dir = vis_dir
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        
        self.vis_writer = SummaryWriter(log_dir=self.log_dir)
    def parse_loss(self):
        """
        Parse loss for training
        """
        for key, value in self.loss.items():
            setattr(self, key, value)


    def configure_data_loader(self):
        """
        Configure data loader for training
        """
        print(f'Configure data loader.')
        
        config = self.opt['dataset']
        dataset = BaseDataset(data_path=config['data_path'], 
                            img_size=config['img_size'], 
                            augmentation=config['augmentation'])
        
        data_loader = DataLoader(dataset=dataset, 
                                batch_size=config['batch_size'], 
                                shuffle=config['shuffle'], 
                                num_workers=config['num_workers'], 
                                pin_memory=config['pin_memory'])
        
        return data_loader


    def configure_net_optimizer(self):
        """
        Configure model for training
        """
        print(f'Configure model.')
        if self.opt.mode == 'vq_vae':
            config = self.opt['network']['vq_vae']
            self.model = VQVAE(in_channels=config['in_channels'], 
                        embedding_dim=config['embedding_dim'], 
                        num_embeddings=config['num_embeddings'], 
                        num_layers=config['num_layers'])
        
        print(f'Configure optimizer.')
        config = self.opt['optimizer']
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=config['lr'], 
                                    betas=config['betas'], 
                                    weight_decay=config['weight_decay'], 
                                    eps=config['eps'])
        
        self.model.to(self.device)
        self.optimizer.to(self.device)


    def configure_loss(self):
        """
        Configure loss for training
        args:
            opt: options for loss
        """
        print(f'Configure loss.')
        
        config = self.opt['loss']
        l1_loss = F.l1_loss(reduction='none')
        perceptual_loss = PerceptualLoss(config['perceptual_loss_type'], 
                                        config['perceptual_loss_layer'], 
                                        config['perceptual_loss_feature_dim'])
        loss = {
            'l1_loss': l1_loss.to(self.device),
            'perceptual_loss': perceptual_loss.to(self.device)
        }
        
        return loss
