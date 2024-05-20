import torch
import numpy as np
import abc
from tqdm import trange, tqdm
import math

from utils.graph_utils import mask_adjs, mask_x, gen_noise, CoM2zero

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, mix, drift_fn):
    super().__init__()
    self.mix = mix
    self.drift_fn = drift_fn

  @abc.abstractmethod
  def update_fn(self, z, t, flags):
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, mix, drift_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.mix = mix
    self.drift_fn = drift_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, z, t, flags):
    pass


# -------- Solve from time 0 to 1 --------
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, mix, drift_fn):
    super().__init__(mix, drift_fn)

  def update_fn(self, feat, pos, flags, edge_mask, t):
    dt = min(1. / self.mix.feat.N, 1 - t[0].item())
    diffusion_feat = self.mix.feat.diffusion(t)
    diffusion_pos = self.mix.pos.diffusion(t)
    drift_feat, drift_pos = self.drift_fn(feat, pos, flags, edge_mask, t) 

    feat_mean = feat + drift_feat * dt
    feat = feat_mean + diffusion_feat[:, None, None] * np.sqrt(dt) * gen_noise(feat, flags)

    pos_mean = pos + drift_pos * dt
    pos_mean = CoM2zero(pos_mean, flags)
    w_pos = gen_noise(pos, flags)
    w_pos = CoM2zero(w_pos, flags)
    pos = pos_mean + diffusion_pos[:, None, None] * np.sqrt(dt) * w_pos
    return feat, feat_mean, pos, pos_mean


class NoneCorrector(object):
  """An empty corrector that does nothing."""
  def __init__(self, mix, drift_fn, snr, scale_eps, n_steps):
    super().__init__()

  def update_fn(self, feat, pos, flags, edge_mask, t):
    return feat, feat, pos, pos


def load_predictor(predictor, mix, drift_fn):
  PREDICTORS = {
    'Euler': EulerMaruyamaPredictor }
  predictor_fn = PREDICTORS[predictor]
  predictor_obj = predictor_fn(mix, drift_fn)
  return predictor_obj


def load_corrector(corrector, mix, drift_fn, snr, scale_eps, n_steps=1):
  CORRECTORS = {
    'None': NoneCorrector}
  corrector_fn = CORRECTORS[corrector]
  corrector_obj = corrector_fn(mix, drift_fn, snr, scale_eps, n_steps)
  return corrector_obj


# -------- PC sampler --------
def get_pc_sampler(mix, shape, sampler, denoise=True, eps=1e-3, device='cuda'):

  def pc_sampler(model, init_flags, prior_samples=None):
    drift_fn = get_drift_fn(mix, model)
    predictor_obj = load_predictor(sampler.predictor, mix, drift_fn)
    corrector_obj = load_corrector(sampler.corrector, mix, drift_fn, 
                                    sampler.snr, sampler.scale_eps, sampler.n_steps)
    feat_conv, pos_conv = [], []
  
    with torch.no_grad():
      # -------- Initial sample --------
      flags = init_flags
      if prior_samples is None:
        feat = mix.feat.bridge(0).prior_sampling(shape.feat, device)
        pos = mix.pos.bridge(0).prior_sampling(shape.pos, device)
      else:
        feat, pos = prior_samples
      feat, pos = mask_x(feat, flags), mask_x(pos, flags)
      pos = CoM2zero(pos, flags)

      edge_mask = make_edge_mask(flags, feat.shape[1])

      steps, T = mix.feat.N, mix.feat.bridge(0).T
      timesteps = torch.linspace(0, T - eps, steps, device=device)

      # -------- Diffusion process --------
      for i in trange(0, (steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape.feat[0], device=t.device) * t

        feat, feat_mean, pos, pos_mean = corrector_obj.update_fn(feat, pos, flags, edge_mask, vec_t)
        feat, feat_mean, pos, pos_mean = predictor_obj.update_fn(feat, pos, flags, edge_mask, vec_t)

        feat_conv.append(feat_mean.detach().cpu())
        pos_conv.append(pos_mean.detach().cpu())
      print(' ')
    return (feat_mean if denoise else feat), (pos_mean if denoise else pos), (feat_conv, pos_conv)
  return pc_sampler


def get_drift_fn(mix, model):
  model.eval()

  def get_drift_from_pred(mix, pred, z, t):
    bridge = mix.bridge(0)
    if 'BB' in mix.bridge_type:
      drift = bridge.drift_time_scaled(t)[:, None, None] * (pred - z)
    elif 'OU' in mix.bridge_type:
      var = bridge.variance(t)
      a_t1 = bridge.a_ou(t, torch.ones_like(t))
      gamma = a_t1 * bridge.a_over_v(t)
      drift = (bridge.alpha_t(t) * var)[:, None, None] * z + \
              (var * gamma)[:, None, None] * (pred / a_t1[:, None, None] - z)
    else:
      raise NotImplementedError(f'Bridge type: {mix.bridge_type} not implemented.')
    return drift

  def drift_fn(feat, pos, flags, edge_mask, t):
    pred_feat, pred_pos = model(feat, pos, flags, edge_mask, t)
    
    drift_feat = get_drift_from_pred(mix.feat, pred_feat, feat, t) 
    drift_pos = get_drift_from_pred(mix.pos, pred_pos, pos, t)
    return drift_feat, drift_pos
    
  return drift_fn

def make_edge_mask(flags, max_node_num):
  adj = torch.ones((flags.size(0), max_node_num, max_node_num), dtype=torch.float).to(flags.device)
  adj = mask_adjs(adj, flags)
  adj[:, range(max_node_num), range(max_node_num)] = 0

  return adj