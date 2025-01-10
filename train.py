import torch
import argparse
from tqdm import tqdm
from collections import OrderedDict
from utils import parse
from utils.train_utils import BaseTrainer


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
            
        
    def _Call_motsync(self):
        """
        Train mode: motsync
        """
        print(f'==> Train mode: {self.mode}')
        # setting torch run.
        config = self.opt['train']
        torch.backends.cudnn.benchmark = True
        self.cur_step, self.cur_epoch = 0, 0
        self.load_pretrained(self.pretrained_path, params=True)
        
        
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
                if self.cur_step % config['print_loss_interval'] == 0:
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
    
    def _Call_gansync(self):
        """
        Train mode: gansync
        """
        pass






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/params.yaml', help='super params to parse.')
    parser.add_argument('--mode', type=str, choices=['vq_vae', 'motsync'], default='motsync')
    args = parser.parse_args()
    
    # load super params.
    print(f'==> Loaded params from {args.cfg}')
    opt = parse(args.cfg)
    
    torch.multiprocessing.set_start_method('spawn')
    # start train.
    T = Trainer(opt, mode=args.mode)
    T._Call_motsync()