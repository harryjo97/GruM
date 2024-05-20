import torch
from utils.graph_utils import mask_x, mask_adjs, gen_noise, CoM2zero
from tqdm import tqdm

def get_pred_3D_loss_fn(mix_feat, mix_pos, train=True, reduce_mean=True, eps=1e-5, loss_type=None):
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: torch.sum(*args, **kwargs)

  def compute_loss(pred, target, loss_coeff):
    losses = torch.square( (pred - target) * loss_coeff[:, None, None] ) 
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * 0.5
    return losses

  def get_loss_coeff(sde, t, loss_type):
    if loss_type=='default':
        loss_coeff = sde.loss_coeff(t)
    elif loss_type=='ones':
        loss_coeff = torch.ones_like(t)
    elif loss_type=='scaled':
        loss_coeff = torch.ones_like(t) * 100
    else:
        raise NotImplementedError(f'Loss type: {loss_type} not implemented.')
    return loss_coeff

  def loss_fn(model, feat, pos, flags, edge_mask, prior_samples=None):
    sde_feat = mix_feat.bridge(feat)
    sde_pos = mix_pos.bridge(pos)
    
    batch_size = feat.size(0)
    t = torch.rand(batch_size, device=feat.device) * (sde_feat.T - eps) # if t=1, the drift term explodes

    # Sampling time 0
    if prior_samples is None:
      feat0 = sde_feat.prior_sampling(feat.shape, feat.device)
      pos0 = sde_pos.prior_sampling(pos.shape, pos.device)
    else:
      feat0, pos0 = prior_samples

    feat0 = mask_x(feat0, flags)
    pos0 = mask_x(pos0, flags)
    pos0 = CoM2zero(pos0, flags)
  
    # Sampling time t 
    mean_feat, std_feat = sde_feat.marginal_prob(feat0, t)
    feat_t = mean_feat + std_feat[:, None, None] * gen_noise(feat, flags)

    mean_pos, std_pos = sde_pos.marginal_prob(pos0, t)
    w_pos = gen_noise(pos, flags)
    w_pos = CoM2zero(w_pos, flags)
    pos_t = mean_pos + std_pos[:, None, None] * w_pos

    # Prediction by NN
    pred_feat, pred_pos = model(feat_t, pos_t, flags, edge_mask, t)

    # Compute Loss
    loss_coeff_feat = get_loss_coeff(sde_feat, t, loss_type.atom_feat)
    loss_coeff_pos  = get_loss_coeff(sde_pos, t, loss_type.pos)

    losses_feat = compute_loss(pred_feat, feat, loss_coeff_feat)
    losses_pos  = compute_loss(pred_pos, pos, loss_coeff_pos)

    losses_feat = losses_feat.mean(dim=0)
    losses_pos  = losses_pos.mean(dim=0)
    losses = losses_feat + losses_pos

    return losses, losses_feat.detach().cpu(), losses_pos.detach().cpu()

  return loss_fn