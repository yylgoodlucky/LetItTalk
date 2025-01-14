import torch
import keyboard
import argparse
from tqdm import tqdm
from collections import OrderedDict
from utils import parse
from utils.train_utils import BaseTrainer

import torch.nn.functional as F


class Trainer(BaseTrainer):
    def __init__(self, 
                 opt: dict, 
                 mode: str, ):
        super(Trainer, self).__init__(opt, mode)
        
        if torch.cuda.device_count() > 1:
            self.device = f'cuda:{opt["gpus"][0]}'
        else:
            self.device = 'cuda'
        
        self.configure_data_loader()
        self.configure_net_optimizer()
        self.configure_loss()
        self.parse_loss()
        self.setup_logger()
        
        
    def set_data(self, data: dict):
        """
        Set data for training with DP support.
        """
        try:
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    # Move tensor data to the specified device
                    value = value.to(self.device, non_blocking=True).to(torch.float32)
                    if torch.isnan(value).any():
                        raise ValueError(f'{key} is nan.')
                setattr(self, key, value)
        except Exception as e:
            print(f"Error in set_data: {str(e)}")
            raise
            
        
    def _Call_motsync(self,
                      init_vis: int = 5):
        """
        Train mode: motsync
        """
        print(f'==> Train mode: {self.mode}')
        # setting torch run.
        config = self.opt['train']
        torch.backends.cudnn.benchmark = True
        self.cur_step, self.cur_epoch = 0, 0
        self.load_params(self.pretrained_path, GAN_model=False)
        
        
        # start training.
        self.model = self.model.to(self.device)
        self.model.train()
        while self.cur_epoch < config['num_epochs']:
            self.cur_epoch += 1
            prog_bar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
            
            cosine_loss = 0
            for step, (data) in prog_bar:
                loss_dict = OrderedDict()
                self.cur_step += 1
                
                self.set_data(data)
                self.optimizer.zero_grad()
                
                # forward
                f_imbed, m_imbed = self.model(self.face_img*self.mouth_img, self.motion_params)
                loss = self.Cosine_loss(f_imbed, m_imbed, self.label)
                
                # backward
                loss.backward()
                self.optimizer.step()
                
                cosine_loss += loss.item()
                
                # update loss
                loss_dict['cosine_loss'] = cosine_loss / (step + 1)
                if self.cur_step % config['print_loss_interval'] == 0 or step < init_vis:
                    print(f'Epoch {self.cur_epoch}, Step {self.cur_step}, Loss: {loss_dict}')
                
                # update logger
                if self.cur_step % config['log_loss_interval'] == 0:
                    self.update_logger(loss_dict, self.cur_step)
                    
                # save ckpt
                if self.cur_step % config['save_ckpt_interval'] == 0:
                    self.save_ckpt(epoch=self.cur_epoch, iteration=self.cur_step)
                    
                # save vis
                if self.cur_step % config['vis_interval'] == 0:
                    pass
                
            # update lr
            self.lr_scheduler.step()
    
    def _Call_gansync(self, 
                      init_vis: int = 5):
        """
        Train mode: gansync
        """
        print(f'==> Train mode: {self.mode}')
        # setting torch run.
        config = self.opt['train']
        torch.backends.cudnn.benchmark = True
        self.cur_step, self.cur_epoch = 0, 0
        self.load_params(self.pretrained_path, GAN_model=True)
        
        # start training.
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.generator.train()
        self.discriminator.train()
        while self.cur_epoch < config['num_epochs']:
            self.cur_epoch += 1
            prog_bar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
            
            d_loss_total, g_loss_total, l1_loss, lpips_loss, id_loss, gv_loss = 0, 0, 0, 0, 0, 0
            for step, (data) in prog_bar:
                loss_dict = OrderedDict()
                self.cur_step += 1
                
                self.set_data(data)
                self.src_al = torch.cat([self.src, self.alpha[:, :1]], dim=1)
                self.masked = self.src * (1 - self.mask)
                
                # 1. 训练判别器
                if self.cur_step > config['adv_loss']:
                    self.optimizer_D.zero_grad()
                    
                    with torch.no_grad():
                        # 生成假样本时不需要计算梯度
                        self.gen = self.generator(self.masked, self.ref, self.motion)
                    
                    fake_pred = self.discriminator(self.gen.detach())  # 注意使用detach()
                    real_pred = self.discriminator(self.src_al)
                    
                    d_loss_real = F.softplus(-real_pred).mean()
                    d_loss_fake = F.softplus(fake_pred).mean()
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    
                    d_loss.backward()
                    self.optimizer_D.step()
                    d_loss_total += d_loss.item()
                    loss_dict['d_loss_total'] = d_loss_total / (step + 1)
                    
                    # gradient penalty
                    use_gp = True
                    if use_gp:  # 需要在配置中添加此选项
                        gp = self.gradient_penalty(self.src_al, self.gen.detach())
                        gp.backward()
                        d_loss = d_loss + 10 * gp
                
                
                # 2. 训练生成器
                self.optimizer_G.zero_grad()
                
                # 重新生成样本
                self.gen = self.generator(self.masked, self.ref, self.motion)    
                
                # import pdb; pdb.set_trace()
                if self.cur_step > config['adv_loss']:
                    fake_pred = self.discriminator(self.gen)
                    g_loss_adv = F.softplus(-fake_pred).mean()
                    g_loss_total += g_loss_adv.item()
                else:
                    g_loss_total = 0.0
                    g_loss_adv = 0.0
                loss_dict['g_loss_total'] = g_loss_total / (step + 1)
                
                
                # Reconstruction loss
                loss_recon_l1 = self.l1_loss(self.gen, self.src_al, reduction='none').mean(dim=(1, 2, 3))
                loss_recon_lpips = self.perceptual_loss(self.gen[:, :3], self.src_al[:, :3])
                
                l1_loss += loss_recon_l1.mean()
                lpips_loss += loss_recon_lpips.mean()
                loss_dict['l1_loss'] = l1_loss / (step + 1)
                loss_dict['lpips_loss'] = lpips_loss / (step + 1)
                
                if self.cur_epoch > 100:
                    loss_recon_id = self.ID_Loss(self.gen[:, :3], self.src_al[:, :3], self.ldmks, self.ldmks, reduction='none')
                    id_loss += loss_recon_id.mean()
                else:
                    loss_recon_id = 0.0
                    id_loss = 0.0
                loss_dict['id_loss'] = id_loss / (step + 1)
                
                
                if self.cur_epoch > 100:
                    loss_GV = self.GV_Loss(self.gen[:, :3], self.src_al[:, :3], reduction='none')
                    gv_loss += loss_GV.mean()
                else:
                    loss_GV = 0.0
                    gv_loss = 0.0
                loss_dict['gv_loss'] = gv_loss / (step + 1)
                
                # Update generator
                loss_recon = loss_recon_l1 + loss_recon_lpips + 0.1 * loss_recon_id + 0.01 * loss_GV
                loss_G = g_loss_adv * 0.1 + loss_recon.mean()
                loss_G.backward()
                self.optimizer_G.step()
                
                # updata loss
                if self.cur_step % config['print_loss_interval'] == 0 or step < init_vis:
                    print(f'Epoch {self.cur_epoch}, Step {self.cur_step}, Loss: {loss_dict}')
                
                # update logger
                if self.cur_step % config['log_loss_interval'] == 0:
                    self.update_logger(loss_dict, self.cur_step)
                
                # save ckpt
                if self.cur_step % config['save_ckpt_interval'] == 0:
                    self.save_ckpt(epoch=self.cur_epoch, iteration=self.cur_step, GAN_model=True)
                
                # save vis
                if self.cur_step % config['vis_interval'] == 0 or step < init_vis:
                    self.save_vis(epoch=self.cur_epoch, 
                                  iteration=self.cur_step, 
                                  img_dict={'masked':self.masked, 
                                            'mask':self.mask, 
                                            'ref':self.ref, 
                                            'gen':self.gen, 
                                            'src':self.src})

            self.lr_scheduler_G.step()
            self.lr_scheduler_D.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/params.yaml', help='super params to parse.')
    parser.add_argument('--mode', type=str, choices=['vq_vae', 'motsync', 'stylesync'], default='motsync')
    args = parser.parse_args()
    
    # load super params.
    print(f'==> Loaded params from {args.cfg}')
    opt = parse(args.cfg)
    
    torch.multiprocessing.set_start_method('spawn')
    # start train.
    T = Trainer(opt, mode=args.mode)
    if args.mode == 'motsync':
        T._Call_motsync()
    elif args.mode == 'stylesync':
        T._Call_gansync()