import os
import time
import pickle
import math
import torch
import yaml

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, \
                         load_ema_from_ckpt, load_sampling_fn

from utils.print_colors import blue, red
from utils.stab_analyze import analyze_valency, DistributionNodes
from utils.graph_utils import save_xyz_file


# -------- Sampler for 3D molecule generation tasks --------
class Sampler_3D_mol(object):
    def __init__(self, config):
        self.config = config
        self.device = load_device()
        self.dataset_info = yaml.load(open('./config/dataset_info_3D.yaml', 'r'), Loader=yaml.FullLoader)
        self.dataset_info = self.dataset_info[config.data.data]

        self.flags_sampler = DistributionNodes(self.dataset_info['n_nodes'])

    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.log_name}')
            start_log(logger, self.configt)
            train_log(logger, self.configt, self.ckpt_dict['params'])
        
        sample_log(logger, self.config)        
        # -------- Load models --------
        self.model = load_model_from_ckpt(self.ckpt_dict['params'], self.ckpt_dict['state_dict'], self.device)
        
        if self.config.sample.use_ema:
            self.ema = load_ema_from_ckpt(self.model, self.ckpt_dict['ema'], self.configt.train.ema)
            self.ema.copy_to(self.model.parameters())

        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        atom_types, positions, flags = [], [], []

        n_samples, batch_size = self.config.sample.n_samples, self.config.sample.batch_size
        num_sampling_rounds = math.ceil(n_samples / batch_size)

        for r in range(num_sampling_rounds):          
            self.init_flags = self.flags_sampler.init_flags(self.config.data.max_node_num, batch_size).to(self.device[0])
            
            feat, pos, (feat_conv, pos_conv) = self.sampling_fn(self.model, self.init_flags)

            if self.configt.data.include_charge:
                feat = feat[:, :, :-1]
            atom_type = feat / self.configt.data.scale.atomtype
            pos = pos / self.configt.data.scale.pos

            atom_types.append(atom_type), positions.append(pos), flags.append(self.init_flags.detach().cpu())
            logger.log(f"{(r+1) * batch_size}/{n_samples} are genearted")

        atom_types = torch.cat(atom_types, dim=0)[:n_samples]
        positions = torch.cat(positions, dim=0)[:n_samples]
        flags = torch.cat(flags, dim=0)[:n_samples]

        # -------- Save generated molecules --------
        if not(os.path.isdir(f'./samples/mols/{self.log_folder_name}')):
            os.makedirs(os.path.join(f'./samples/mols/{self.log_folder_name}'))
            
        with open(os.path.join('samples', 'mols', f'{self.log_folder_name}', f'{self.log_name}.txt'), 'w') as f:
            save_xyz_file(f, atom_types, positions, flags, self.dataset_info)   
            

        
        # -------- Evaluation --------
        validity_dict, rdkit_metrics = analyze_valency(atom_types, positions, flags, self.dataset_info)     
        
        fraction_mol_stable = validity_dict['mol_stable']
        fraction_atm_stable = validity_dict['atm_stable']
        fraction_connected_mol = validity_dict['connected_mol']
        logger.log(f"Num Mol: {n_samples} | Mol Stability: {fraction_mol_stable * 100:.2f} | Atom Stability: {fraction_atm_stable * 100:.2f} | Connected : {fraction_connected_mol * 100:.2f}")
        logger.log('='*100)