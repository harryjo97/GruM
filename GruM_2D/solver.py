import torch
import numpy as np
import abc
from tqdm import trange, tqdm
import math

from utils.graph_utils import mask_adjs, mask_x, gen_noise

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

  def update_fn(self, x, adj, y, flags, t):
    dt = min(1. / self.mix.adj.N, 1 - t[0].item())
    diffusion_x = self.mix.x.diffusion(t)
    diffusion_adj = self.mix.adj.diffusion(t)
    drift_x, drift_adj = self.drift_fn(x, adj, y, flags, t) 

    x_mean = x + drift_x * dt
    x = x_mean + diffusion_x[:, None, None] * np.sqrt(dt) * gen_noise(x, flags, sym=False)
    adj_mean = adj + drift_adj * dt
    adj = adj_mean + diffusion_adj[:, None, None] * np.sqrt(dt) * gen_noise(adj, flags)
    return x, x_mean, adj, adj_mean


class NoneCorrector(object):
  """An empty corrector that does nothing."""
  def __init__(self, mix, drift_fn, snr, scale_eps, n_steps):
    super().__init__()

  def update_fn(self, x, adj, y, flags, t):
    return x, x, adj, adj


class LangevinCorrector(Corrector):
  """A Langevin-like corrector. Only used for Planar dataset.
  """
  def __init__(self, mix, drift_fn, snr, scale_eps, n_steps):
    super().__init__(mix, drift_fn, snr, scale_eps, n_steps)

  # -------- Use un-scaled drift & diffusion --------
  def score_from_drift(self, mix, drift, t):
    """Use the drift as the score.
    """
    diffusion = mix.diffusion(t)
    time_scaled = 1./(diffusion**2) 
    return drift * time_scaled[:,None,None]

  def correct_one_step(self, mix, drift, noise, z, t):
    alpha = torch.ones_like(t)
    grad = self.score_from_drift(mix, drift, t)
    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
    noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
    step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
    z_mean = z + step_size[:, None, None] * grad
    z = z_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * self.scale_eps
    return z, z_mean

  def update_fn(self, x, adj, y, flags, t):
    for i in range(self.n_steps):
      drift_x, drift_adj = self.drift_fn(x, adj, y, flags, t)
      noise_x = gen_noise(x, flags, sym=False)
      x, x_mean = self.correct_one_step(self.mix.x, drift_x, noise_x, x, t)
      noise_adj = gen_noise(adj, flags, sym=True)
      adj, adj_mean = self.correct_one_step(self.mix.adj, drift_adj, noise_adj, adj, t)
    return x, x_mean, adj, adj_mean


def load_predictor(predictor, mix, drift_fn):
  PREDICTORS = {
    'Euler': EulerMaruyamaPredictor }
  predictor_fn = PREDICTORS[predictor]
  predictor_obj = predictor_fn(mix, drift_fn)
  return predictor_obj


def load_corrector(corrector, mix, drift_fn, snr, scale_eps, n_steps=1):
  CORRECTORS = {
    'None': NoneCorrector,
    'Langevin': LangevinCorrector }
  corrector_fn = CORRECTORS[corrector]
  corrector_obj = corrector_fn(mix, drift_fn, snr, scale_eps, n_steps)
  return corrector_obj


# -------- PC sampler --------
def get_pc_sampler(mix, shape, sampler, denoise=True, eps=1e-3, device='cuda'):

  def pc_sampler(model, init_flags, prior_samples=None):
    drift_fn = get_drift_fn(mix, model)
    predictor_obj = load_predictor(sampler.predictor, mix, drift_fn)
    corrector_obj = load_corrector(sampler.corrector, mix, drift_fn, sampler.snr, 
                                    sampler.scale_eps, sampler.n_steps)
    x_conv = []
    adj_conv = []
    with torch.no_grad():
      # -------- Initial sample --------
      flags = init_flags
      if prior_samples is None:
        x = mix.x.bridge(0).prior_sampling(shape.x, device)
        adj = mix.adj.bridge(0).prior_sampling_sym(shape.adj, device)
      else:
        x, adj = prior_samples
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      steps, T = mix.adj.N, mix.adj.bridge(0).T
      timesteps = torch.linspace(0, T - eps, steps, device=device)

      # -------- Diffusion process --------
      for i in trange(0, (steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape.adj[0], device=t.device) * t
        y = vec_t.unsqueeze(-1)

        x, x_mean, adj, adj_mean = corrector_obj.update_fn(x, adj, y, flags, vec_t)
        x, x_mean, adj, adj_mean = predictor_obj.update_fn(x, adj, y, flags, vec_t)

        x_conv.append(x_mean.detach().cpu())
        adj_conv.append(adj_mean.detach().cpu())
      print(' ')
    return (x_mean if denoise else x), (adj_mean if denoise else adj), (x_conv, adj_conv)
  return pc_sampler


def get_drift_fn(mix, model):
  model.eval()

  def get_drift_from_pred(mix, pred, z, t):
    drift = pred
    bridge = mix.bridge(0)
    if 'BB' in mix.bridge_type:
      drift = bridge.drift_time_scaled(t)[:, None, None] * (drift - z)
    elif 'OU' in mix.bridge_type:
      var = bridge.variance(t)
      a_t1 = bridge.a_ou(t, torch.ones_like(t))
      gamma = var * a_t1 * bridge.a_over_v(t)
      drift = (bridge.alpha_t(t) * var)[:, None, None] * z + \
              gamma[:, None, None] * (drift / a_t1[:, None, None] - z)
    else:
      raise NotImplementedError(f'Bridge type: {mix.bridge_type} not implemented.')
    return drift

  def drift_fn(x, adj, y, flags, t):
    pred_x, pred_adj = model(x, adj, y, flags) 
    drift_x = get_drift_from_pred(mix.x, pred_x, x, t)
    drift_adj = get_drift_from_pred(mix.adj, pred_adj, adj, t)
    return drift_x, drift_adj
    
  return drift_fn