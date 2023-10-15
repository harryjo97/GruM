import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.print_colors import red
from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
                         load_ema, load_mix_3D_loss_fn, load_ema_from_ckpt, \
                         load_ckpt, load_opt_from_ckpt, load_model_from_ckpt
from utils.logger import Logger, set_log, start_log, train_log

class Trainer3D(object):
    def __init__(self, config):
        super(Trainer3D, self).__init__()
        self.config = config
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        
        self.train_loader, self.test_loader = load_data(self.config)

        self.params = load_model_params(self.config)
    def train(self, ts):
        self.config.exp_name = ts

        # -------- Load models, optimizers, ema --------
        if not self.config.train.resume:
            self.ckpt = f'{ts}'
            self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

            self.model, self.optimizer, self.scheduler = load_model_optimizer(self.params, self.config.train, 
                                                                                self.device)
            self.ema = load_ema(self.model, decay=self.config.train.ema)
            start_epoch = 0
        else:
            if self.config.ckpt is None:
                raise ValueError("To resume the training, specify the checkpoint path.")
            self.ckpt = self.config.ckpt + '_resume'

            self.ckpt_dict = load_ckpt(self.config, self.device)
            self.config = self.ckpt_dict['config']
            self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)
        
            self.model = load_model_from_ckpt(self.ckpt_dict['params'], self.ckpt_dict['state_dict'], self.device)
            self.optimizer = load_opt_from_ckpt(self.config.train, self.ckpt_dict['optimizer'], self.model)
            self.ema = load_ema_from_ckpt(self.model, self.ckpt_dict['ema'], self.config.train.ema)
            
            start_epoch = self.ckpt_dict['epoch'] + 1

        print(red(f'{self.ckpt}'))

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config, self.params)

        if self.config.train.use_tensorboard:
            writer = SummaryWriter(os.path.join(*['logs_train', 'tensorboard', self.config.data.data, 
                                                self.config.train.name, self.config.exp_name]))

        self.loss_fn = load_mix_3D_loss_fn(self.config)

        # -------- Training --------
        for epoch in trange(start_epoch, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):

            self.train_feat = []
            self.train_pos = []
            t_start = time.time()

            self.model.train()

            for _, train_b in enumerate(self.train_loader):
                train_b = train_b.to(f'cuda:{self.device[0]}')
                self.optimizer.zero_grad()

                loss, loss_feat, loss_pos = self.loss_fn(self.model, 
                    train_b.node_feat, train_b.positions, train_b.flags, train_b.edge_mask)

                loss.backward()

                if self.config.train.grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)
                else:
                    grad_norm = 0

                self.optimizer.step()
                
                # -------- EMA update --------
                self.ema.update(self.model.parameters())
                
                self.train_feat.append(loss_feat.cpu().detach().item())
                self.train_pos.append(loss_pos.cpu().detach().item())

            if self.config.train.lr_schedule:
                self.scheduler.step()

            mean_train_feat = np.mean(self.train_feat)
            mean_train_pos = np.mean(self.train_pos)
    
            if self.config.train.use_tensorboard:
                writer.add_scalar("train_feat", mean_train_feat, epoch)
                writer.add_scalar("train_pos", mean_train_pos, epoch)
                writer.flush()

            # -------- Log losses --------
            logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                        f'train feat: {mean_train_feat:.3e} | '
                        f'train pos: {mean_train_pos:.3e} | '
                        f'grad_norm: {grad_norm:.2e} |', verbose=False)

            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''
        
                torch.save({ 
                    'epoch': epoch,
                    'config': self.config,
                    'params' : self.params,
                    'state_dict': self.model.state_dict(), 
                    'ema': self.ema.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, f'./checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth')
            
            if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                tqdm.write(f'[EPOCH {epoch+1:04d}] train feat: {mean_train_feat:.3e} | train pos: {mean_train_pos:.3e} |')
        print(' ')
        if self.config.train.use_tensorboard:
            writer.close()
        return self.ckpt
        