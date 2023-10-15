import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import math


# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs

# -------- Generate noise --------
def gen_noise(x, flags):
    z = torch.randn_like(x)
    z = mask_x(z, flags)
    return z

# -------- Move center of mass of given positions to zero --------
def CoM2zero(x, flags):
    mean = (x.sum(dim=1) / flags.sum(dim=1, keepdim=True)).unsqueeze(1)
    x = mask_x(x - mean, flags)
    return x

# -------- Save generated samples --------
def save_xyz_file(f, one_hot, positions, flags, dataset_info):
    atomsxmol = torch.sum(flags, dim=1)

    for batch_i in range(one_hot.size(0)):
        f.write("Sample %d\n\n" % batch_i)
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        n_atoms = int(atomsxmol[batch_i])
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = dataset_info['atom_decoder'][atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.write('='*100)
        f.write('\n')
