import os
import torch
import torchvision
import importlib
import torch.nn.functional as F
from torch.utils.data import DataLoader
from os.path import join

from models.vae import *
from models.motnet import *
from models.stylegan.LIA import *
from utils.loss_utils import *
from utils.Identity import IdentityLoss

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
                                        data_file=self.opt['data_file'],
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
            self.load_pretrained(self.pretrained_path, GAN_model=False)
            
            print(f'==> Configure optimizer.')
            config = self.opt['optim']['optimizer_g']
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=float(config['lr']), 
                                        betas=list(map(float, config['betas'])), 
                                        weight_decay=float(config['weight_decay']), 
                                        eps=float(config['eps']))
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                        step_size=int(config['step_size']), 
                                        gamma=float(config['gamma']))
            self.model.to(self.device)
            
            # multi-gpu
            if len(self.opt['gpus']) > 1:
                print(f'==> use gpu {self.opt["gpus"]}.')
                self.model = torch.nn.DataParallel(self.model, device_ids=[int(i) for i in self.opt['gpus']])
            
            
        if self.mode == 'stylesync':
            self.generator = Generator(
                                size=config['generator']['size'],
                                style_dim=config['generator']['style_dim'],
                                motion_dim=config['generator']['motion_dim'],
                                channel_multiplier=config['generator']['channel_multiplier'],
                                device_id=self.opt['gpus'][0])
            self.discriminator = Discriminator(
                                size=config['discriminator']['size'],
                                in_channel=config['discriminator']['in_channel'])
            self.load_pretrained(self.pretrained_path, GAN_model=True)
            
            print(f'==> Configure optimizer.')
            config_g = self.opt['optim']['optimizer_g']
            config_d = self.opt['optim']['optimizer_d']
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), 
                                        lr=float(config_g['lr']), 
                                        betas=list(map(float, config_g['betas'])), 
                                        weight_decay=float(config_g['weight_decay']), 
                                        eps=float(config_g['eps']))
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), 
                                        lr=float(config_d['lr']), 
                                        betas=list(map(float, config_d['betas'])), 
                                        weight_decay=float(config_d['weight_decay']), 
                                        eps=float(config_d['eps']))
            self.lr_scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, 
                                        step_size=int(config_g['step_size']), 
                                        gamma=float(config_g['gamma']))
            self.lr_scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, 
                                        step_size=int(config_d['step_size']), 
                                        gamma=float(config_d['gamma']))
            
            self.generator.to(self.device)
            self.discriminator.to(self.device)
            
            # multi-gpu
            if len(self.opt['gpus']) > 1:
                print(f'==> use gpu {self.opt["gpus"]}.')
                self.generator = torch.nn.DataParallel(self.generator, device_ids=[int(i) for i in self.opt['gpus']])
                self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=[int(i) for i in self.opt['gpus']])
        
        


    def configure_loss(self):
        """
        Configure loss for training
        args:
            opt: options for loss
        """
        print(f'==> Configure loss.')
        
        config = self.opt['loss']
        
        loss_dict = {}
        for loss in config.keys():
            if loss == 'l1_loss':
                l1_loss = L1_loss()
                loss_dict[loss] = l1_loss.to(self.device)
            elif loss == 'perceptual_loss':
                perceptual_loss = LPIPSLoss()
                loss_dict[loss] = perceptual_loss.to(self.device)
            elif loss == 'Cosine_loss':
                cosine_loss = Cosine_loss()
                loss_dict[loss] = cosine_loss.to(self.device)
            elif loss == 'GV_Loss':
                gv_loss = GradientVariance(patch_size=config['GV_Loss']['patch_size'])
                loss_dict[loss] = gv_loss.to(self.device)
            elif loss == 'Identity_Loss':
                id_loss = IdentityLoss()
                loss_dict[loss] = id_loss.to(self.device)
        
        self.loss = loss_dict
        
        # ------ loss configure end ------

    def parse_loss(self):
        """
        Parse loss for training
        """
        for key, value in self.loss.items():
            setattr(self, key, value)
            
        # ------ loss parse end ------
        
        
    def save_ckpt(self, epoch: int, 
                  iteration: int, 
                  GAN_model: bool = False):
        """
        Save checkpoint for training
        Args:
            epoch (int, optional): Current epoch number
            iteration (int, optional): Current iteration number
        """
        # Save model state
        if GAN_model:
            state = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'scheduler_G': self.lr_scheduler_G.state_dict(),
            'scheduler_D': self.lr_scheduler_D.state_dict(),
            'epoch': epoch,
            'iteration': iteration
        }
        else:
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
        print(f'==> Latest model is saved in {save_path}')
    
    
    def save_vis(self, 
                 epoch: int, 
                 iteration: int, 
                 img_dict: dict):
        """
        Save visualization for training
        Args:
            epoch (int, optional): Current epoch number
            iteration (int, optional): Current iteration number
            img_dict (dict): Dictionary of images to be visualized
        """
        with torch.no_grad():
            # Process images (convert to range [0, 1])
            masked = (img_dict['masked'].clamp(-1, 1) + 1) / 2
            mask = img_dict['mask'].clamp(-1, 1)
            ref = (img_dict['ref'].clamp(-1, 1) + 1) / 2
            gen = (img_dict['gen'][:, :3].clamp(-1, 1) + 1) / 2
            src = (img_dict['src'].clamp(-1, 1) + 1) / 2
            
            # Take only first 4 images
            n_samples = min(5, gen.shape[0])
            
            # Create grid of images
            grid = torchvision.utils.make_grid(
                torch.cat([
                    masked[:n_samples],
                    mask[:n_samples],
                    ref[:n_samples],
                    gen[:n_samples],
                    src[:n_samples],
                ], dim=0),
                nrow=n_samples,  # Number of images per row
                padding=2,
                normalize=False
            )
            
            # Save grid
            save_path = os.path.join(self.vis_dir, f'epoch_{epoch}_step_{iteration}.png')
            torchvision.utils.save_image(grid, save_path)
    
    
    
    def load_pretrained(self, 
                   pretrained_path: str, 
                   GAN_model: bool = False):
        """
        Load pretrained model
        """
        if not os.path.exists(pretrained_path):
            print(f'==> Pretrained model not found in {pretrained_path}.')
            return
        
        state_dict = torch.load(pretrained_path, map_location=self.device)
        
        if GAN_model:
            # Handle DataParallel state dict
            model_g_state_dict = state_dict['generator']
            model_d_state_dict = state_dict['discriminator']
            new_g_new = {}
            for k, v in model_g_state_dict.items():
                new_g_new[k.replace('module.', '')] = v
            self.generator.load_state_dict(new_g_new, strict=True)
            
            new_d_new = {}
            for k, v in model_d_state_dict.items():
                new_d_new[k.replace('module.', '')] = v
            self.discriminator.load_state_dict(new_d_new, strict=True)
                
        else:
            # Handle DataParallel state dict
            model_state_dict = state_dict['model']
            new_s = {}
            for k, v in model_state_dict.items():
                new_s[k.replace('module.', '')] = v
            self.model.load_state_dict(new_s, strict=True)
            
        print(f'==> Load pretrained model from {pretrained_path}.')
    
    
    def load_params(self, 
                   pretrained_path: str, 
                   GAN_model: bool = False, 
                   overwrite_params: bool = True):
        """
        Load pretrained params
        """
        if not os.path.exists(pretrained_path):
            print(f'==> Pretrained model not found in {pretrained_path}.')
            return
        
        state_dict = torch.load(pretrained_path, map_location=self.device)
        
        if GAN_model:
            self.cur_epoch = state_dict['epoch']
            self.cur_step = state_dict['iteration']
            self.optimizer_G.load_state_dict(state_dict['optimizer_G'])
            self.optimizer_D.load_state_dict(state_dict['optimizer_D'])
            self.lr_scheduler_G.load_state_dict(state_dict['scheduler_G'])
            self.lr_scheduler_D.load_state_dict(state_dict['scheduler_D'])
        else:
            self.cur_epoch = state_dict['epoch']
            self.cur_step = state_dict['iteration']
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.lr_scheduler.load_state_dict(state_dict['scheduler'])
            
        print(f'==> Load pretrained params from {pretrained_path}.')
        
    
    def gradient_penalty(self, real_samples, fake_samples):
        """计算 WGAN-GP 的梯度惩罚项"""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        
        # 在真实样本和生成样本之间进行插值
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # 计算判别器对插值样本的输出
        d_interpolates = self.discriminator(interpolates)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # 计算梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty