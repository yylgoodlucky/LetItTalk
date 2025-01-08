import torch
import argparse

from utils import parse
from utils.train_utils import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, 
                 opt: dict, 
                 mode: str,
                 device: str = 'cuda'):
        super(Trainer, self).__init__(opt, mode, device)
        
        self.configure_data_loader()
        self.configure_net_optimizer()
        self.configure_loss()
        
        self.parse_loss()
        self.setup_logger()
        
    def __call__(self):
        print(f'Train mode: {self.mode}')
        
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='super params to parse.')
    parser.add_argument('--mode', type=str, choices=['vq_vae'], default='vq_vae')
    args = parser.parse_args()
    
    # load super params.
    print(f'Loaded params from {args.cfg}')
    opt = parse(args.cfg)
    
    torch.multiprocessing.set_start_method('spawn')
    # start train.
    trainer = Trainer(opt, mode=args.mode)
    trainer()