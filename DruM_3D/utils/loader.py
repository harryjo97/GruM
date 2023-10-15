import torch
import torch_geometric
import random
import numpy as np
import math
from easydict import EasyDict as edict

from models.E3_GNN import EGNN

from mix import DiffusionMixture

from losses import get_pred_3D_loss_fn
from solver import get_pc_sampler
from utils.ema import ExponentialMovingAverage

def load_seed(seed):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def load_device():
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = 'cpu'
    return device


def load_model(params):
    params_ = params.copy()
    model_type = params_.pop('model_type', None)
    if model_type == 'EGNN':
        model = EGNN(**params_)
    else:
        raise ValueError(f"Model Name <{model_type}> is Unknown")
    return model


def load_model_optimizer(params, config_train, device, resume_state_dict=None):
    model = load_model(params)
    if resume_state_dict is not None:
        if 'module.' in list(resume_state_dict.keys())[0]:
            resume_state_dict = {k[7:]: v for k, v in resume_state_dict.items()}
        model.load_state_dict(resume_state_dict)

    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')

    if config_train.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.lr, 
                                    weight_decay=config_train.weight_decay, amsgrad=True)
    else:
        raise NotImplementedError(f'Optimizer:{config_train.optimizer} not implemented.')
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config_train.lr_decay)
    
    return model, optimizer, scheduler


def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(model, ema_state_dict, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(config, get_graph_list=False):
    if config.data.data in ['QM9_3D', 'GEOM_DRUGS']:
        from utils.data_loader_mol_3D import dataloader
        return dataloader(config)
    else:
        raise NotImplementedError()


def load_batch(batch, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    return x_b, adj_b


def load_mix(config_mix):
    mix_type = config_mix.type
    drift_coeff = config_mix.drift_coeff
    sigma_0 = config_mix.sigma_0
    sigma_1 = config_mix.sigma_1
    num_scales = config_mix.num_scales
    mix = DiffusionMixture(bridge=mix_type, drift_coeff=drift_coeff,
                            sigma_0=sigma_0, sigma_1=sigma_1, N=num_scales)
    return mix


def load_mix_3D_loss_fn(config):
    reduce_mean = config.train.reduce_mean
    mix_feat = load_mix(config.mix.atom_feat)
    mix_pos = load_mix(config.mix.pos)

    get_loss_fn = get_pred_3D_loss_fn

    loss_fn = get_loss_fn(mix_feat, mix_pos, train=True, 
                        reduce_mean=config.train.reduce_mean, eps=config.train.eps, loss_type=config.train.loss_type)

    return loss_fn


def load_sampling_fn(config_train, config_sampler, config_sample, device):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    get_sampler = get_pc_sampler

    mix = edict({'feat': load_mix(config_train.mix.atom_feat), 'pos': load_mix(config_train.mix.pos)})

    max_node_num  = config_train.data.max_node_num
    batch_size = config_sample.batch_size
    
    shape = edict({'feat': (batch_size, max_node_num, config_train.data.max_feat_num+int(config_train.data.include_charge)),
                   'pos': (batch_size, max_node_num, 3)})
    sampling_fn = get_sampler(mix=mix, shape=shape, sampler=config_sampler,
                                denoise=config_sample.noise_removal, 
                                eps=config_sample.eps, device=device_id)
    return sampling_fn


def load_model_params(config):
    cfg = config.model
    max_feat_num = config.data.max_feat_num
    max_node_num = config.data.max_node_num
    include_charge = config.data.include_charge

    params = {'model_type': cfg.type, 'max_feat_num': max_feat_num, 
                'n_layers': cfg.num_layers, 'nhid': cfg.nhid, 
                'time_cond': cfg.time_cond, 'include_charge':include_charge, 
                'max_node_num':max_node_num}
    return params


def load_ckpt(config, device, ts=None, return_ckpt=False):
    device_id = f'cuda:{device[0]}' if isinstance(device, list) else device
    path = f'checkpoints/{config.data.data}/{config.ckpt}.pth'
    ckpt_dict = torch.load(path, map_location=device_id)
    print(f'{path} loaded')
    return ckpt_dict


def load_model_from_ckpt(params, state_dict, device):
    model = load_model(params)
    if 'module.' in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f'cuda:{device[0]}')
    return model


def load_opt_from_ckpt(config_train, state_dict, model):
    if config_train.optimizer=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.lr, amsgrad=True,
                                        weight_decay=config_train.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer:{config_train.optimizer} not implemented.')
    optimizer.load_state_dict(state_dict)
    return optimizer
