import os
import torch
import importlib
import torch.nn.functional as F
from torch.utils.data import DataLoader
from os.path import join

from models.vae import *
from models.motnet import *
from models.stylegan.LIA import *
from utils.loss_utils import *

from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self, 
                 opt: dict, 
                 mode: str, 
                 device: str = 'cuda'):
        super(BaseTrainer, self).__init__()
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


    def update_logger(self, loss_dict: dict, step: int):
        """
        Update logger for training
        """
        for key, value in loss_dict.items():
            self.vis_writer.add_scalar(key, value, step)

    
    def configure_data_loader(self):
        """
        Configure data loader for training
        """
        print(f'==> Configure data loader.')
        
        config = self.opt['dataset']
        
        dataset_module = importlib.import_module(f'data.{self.mode.lower()}_dataset')
        dataset = dataset_module.DataGen(
                                        data_path=self.opt['data_path'], 
                                        **config)
        
        self.data_loader = DataLoader(dataset=dataset, 
                                batch_size=config['batch_size'], 
                                shuffle=config['shuffle'], 
                                num_workers=config['num_workers'], 
                                pin_memory=config['pin_memory'])
        
        # ------ data loader configure end ------

    def configure_net_optimizer(self):
        """
        Configure model for training
        """
        print(f'==> Configure model.')
        config = self.opt['network'][self.mode]
        self.pretrained_path = config['pretrained']
        
        if self.mode == 'vq_vae':
            self.model = VQVAE(in_channels=config['in_channels'], 
                        embedding_dim=config['embedding_dim'], 
                        num_embeddings=config['num_embeddings'], 
                        num_layers=config['num_layers'])
            
        if self.mode == 'motsync':
            self.model = MotSync(opts=config, 
                                 n_mlp=config['n_mlp'], 
                                 embeding_dim=config['embeding_dim'])
            
            self.load_pretrained(self.pretrained_path)
        
        if self.mode == 'stylesync':
            self.generator = Generator(opts=config, 
                                 n_mlp=config['n_mlp'], 
                                 embeding_dim=config['embeding_dim'])
            
            self.discriminator = Discriminator(opts=config, 
                                 n_mlp=config['n_mlp'], 
                                 embeding_dim=config['embeding_dim'])
        
        # multi-gpu
        if len(self.opt['gpus']) > 1:
            print(f'==> use gpu {self.opt["gpus"]}.')
            self.model = torch.nn.DataParallel(self.model, device_ids=[int(i) for i in self.opt['gpus']])
        
        # ------ model configure end ------

        print(f'==> Configure optimizer.')
        config = self.opt['optimizer']
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=float(config['lr']), 
                                    betas=list(map(float, config['betas'])), 
                                    weight_decay=float(config['weight_decay']), 
                                    eps=float(config['eps']))
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                        step_size=int(config['step_size']), 
                                                        gamma=float(config['gamma']))
        
        self.model.to(self.device)
        
        # ------ optimizer configure end ------

    def configure_loss(self):
        """
        Configure loss for training
        args:
            opt: options for loss
        """
        print(f'==> Configure loss.')
        
        config = self.opt['loss']
        
        loss_dict = {}
        for loss in config['loss_type']:
            if loss == 'l1_loss':
                l1_loss = F.l1_loss(reduction='none')
                loss_dict[loss] = l1_loss.to(self.device)
            elif loss == 'perceptual_loss':
                perceptual_loss = PerceptualLoss(config['perceptual_loss_type'], 
                                        config['perceptual_loss_layer'], 
                                        config['perceptual_loss_feature_dim'])
                loss_dict[loss] = perceptual_loss.to(self.device)
            elif loss == 'Cosine_loss':
                cosine_loss = Cosine_loss()
                loss_dict[loss] = cosine_loss.to(self.device)
        
        self.loss = loss_dict
        
        # ------ loss configure end ------

    def parse_loss(self):
        """
        Parse loss for training
        """
        for key, value in self.loss.items():
            setattr(self, key, value)
            
        # ------ loss parse end ------
        
        
    def save_ckpt(self, epoch=None, iteration=None):
        """
        Save checkpoint for training
        Args:
            epoch (int, optional): Current epoch number
            iteration (int, optional): Current iteration number
        """
        # Save model state
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'iteration': iteration
        }
        
        # Save latest checkpoint
        save_filename = f'{self.mode}_latest.pth'
        save_path = join(self.ckpt_dir, save_filename)
        torch.save(state, save_path)
    
    
    def save_vis(self, epoch=None, iteration=None):
        """
        Save visualization for training
        Args:
            epoch (int, optional): Current epoch number
            iteration (int, optional): Current iteration number
        """
        pass
    
    
    def load_pretrained(self, pretrained_path: str, params: bool = False):
        """
        Load pretrained model
        """
        state_dict = torch.load(pretrained_path, map_location=self.device)
        
        if params:
            self.cur_epoch = state_dict['epoch']
            self.cur_step = state_dict['iteration']
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.lr_scheduler.load_state_dict(state_dict['scheduler'])
        else:
            # Handle DataParallel state dict
            model_state_dict = state_dict['model']
            if len(self.opt['gpus']) > 1:  # If using DataParallel
                if not list(model_state_dict.keys())[0].startswith('module.'):
                    model_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}
            else:  # If not using DataParallel
                if list(model_state_dict.keys())[0].startswith('module.'):
                    model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            
            # 使用严格模式加载模型权重
            try:
                self.model.load_state_dict(model_state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: {str(e)}")
                print("Attempting to load with strict=False...")
                # 如果严格模式失败，尝试非严格模式
                self.model.load_state_dict(model_state_dict, strict=False)
            
        print(f'==> Load pretrained model from {pretrained_path}.')