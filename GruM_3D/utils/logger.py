import os

class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str, verbose=True):
        if self.lock:
            self.lock.acquire()
        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)
        if self.lock:
            self.lock.release()
        if verbose:
            print(str)


def set_log(config, is_train=True):
    data = config.data.data
    exp_name = config.train.name

    log_folder_name = os.path.join(*[data, exp_name])
    root = 'logs_train' if is_train else 'logs_sample'
    if not(os.path.isdir(f'./{root}/{log_folder_name}')):
        os.makedirs(os.path.join(f'./{root}/{log_folder_name}'))
    log_dir = os.path.join(f'./{root}/{log_folder_name}/')

    if not(os.path.isdir(f'./checkpoints/{data}')) and is_train:
        os.makedirs(os.path.join(f'./checkpoints/{data}'))
    ckpt_dir = os.path.join(f'./checkpoints/{data}/')

    print('-'*100)
    print("Make Directory {} in Logs".format(log_folder_name))

    return log_folder_name, log_dir, ckpt_dir


def check_log(log_folder_name, log_name):
    return os.path.isfile(f'./logs_sample/{log_folder_name}/{log_name}.log')


def data_log(logger, config):
    logger.log(f'[{config.data.data}]   init={config.data.init} ({config.data.max_feat_num})   seed={config.seed}   '
                f'batch_size={config.data.batch_size}   scale=({config.data.scale.atomtype}, {config.data.scale.charge}, {config.data.scale.pos})   ')

def mix_log(logger, config_mix):
    mix_feat = config_mix.atom_feat
    mix_pos = config_mix.pos

    if mix_feat.type == 'BB':
        logger.log(f'(feat  :{mix_feat.type}), [{mix_feat.sigma_0:.3e},{mix_feat.sigma_1:.3e}]) N={mix_feat.num_scales}') 
    else:
        logger.log(f'(feat  :{mix_feat.type}), {mix_feat.drift_coeff}, '
                    f'[{mix_feat.sigma_0:.3e},{mix_feat.sigma_1:.3e}]) N={mix_feat.num_scales})') 

    if mix_pos.type == 'BB':
        logger.log(f'(pos   :{mix_pos.type}), [{mix_pos.sigma_0:.3e},{mix_pos.sigma_1:.3e}]) N={mix_pos.num_scales}') 
    else:
        logger.log(f'(pos   :{mix_pos.type}), {mix_pos.drift_coeff}, '
                    f'[{mix_pos.sigma_0:.3e},{mix_pos.sigma_1:.3e}]) N={mix_pos.num_scales})')                     
    logger.log('-'*100)


def model_log(logger, params):
    logger.log(f"{params['model_type']}    : nhid={params['nhid']} layer={params['n_layers']} "
                f"include_charge={params['include_charge']} time_cond={params['time_cond']}")
    logger.log('-'*100)


def start_log(logger, config):
    logger.log('-'*100)
    data_log(logger, config)
    logger.log('-'*100)


def train_log(logger, config, params):
    logger.log(f'lr={config.train.lr:.1e} schedule={config.train.lr_schedule}  '
                f'epochs={config.train.num_epochs} optimizer={config.train.optimizer} '
                f'weight_decay={config.train.weight_decay} grad_norm={config.train.grad_norm} ')
    logger.log(f'eps={config.train.eps:.1e} ema={config.train.ema} loss_type={config.train.loss_type}')
    logger.log('-'*100)
    mix_log(logger, config.mix)
    model_log(logger, params)


def sample_log(logger, config):
    sampler = config.sampler
    sample_log = f"({sampler.predictor})+({sampler.corrector}) " 
    if sampler.corrector != 'None':
        sample_log += f'|| snr={sampler.snr:.2f} seps={sampler.scale_eps:.1f} n_steps={sampler.n_steps} '
    sample_log += f"eps={config.sample.eps} ema={config.sample.use_ema}"
    logger.log(sample_log)
    logger.log(f"Num Samples={config.sample.n_samples}  Batch size={config.sample.batch_size}")
    logger.log('-'*100)