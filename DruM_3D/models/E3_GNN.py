import torch
import torch.nn.functional as F
import torch.nn as nn

# from models.layers import DenseGCNConv, MLP
from models.E3_layers import E3Block, coord2diff

class EGNN(torch.nn.Module):
    def __init__(self, max_feat_num, n_layers, nhid, max_node_num, include_charge=True, time_cond=False):
        super().__init__()
        self.nfeat = max_feat_num + int(include_charge)
        self.n_layers = n_layers
        self.nhid = nhid
        self.time_cond = time_cond
        self.include_charge = include_charge
        self.max_N = max_node_num

        # Fully connected edge index
        adj = torch.ones((self.max_N, self.max_N))
        self.full_edge_index = adj.nonzero(as_tuple=False).T

        self.embedding_in = nn.Linear(self.nfeat + int(self.time_cond), self.nhid)
        self.embedding_out = nn.Linear(self.nhid, self.nfeat)

        self.layers = torch.nn.ModuleList()
        coords_range = float(15 / self.n_layers)
        for _ in range(self.n_layers):
            self.layers.append(E3Block(self.nhid, coords_range))
    
    def forward(self, h, x, flags, edge_mask, t=None):
        x_ = x
        bs, n_nodes, _ = h.shape
        h = h.view(bs * n_nodes, -1)
        x = x.view(bs * n_nodes, -1)
        flags = flags.view(bs * n_nodes, -1)
        
        edge_index = self.make_edge_index(bs).to(h.device)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, -1)

        d, _ = coord2diff(x, edge_index)

        if self.time_cond:
            t = t.view(bs, 1).repeat(1, n_nodes).view(bs * n_nodes, 1)
            h = torch.cat([h, t], dim=1)

        h = self.embedding_in(h) * flags
        for layer in self.layers:
            h, x = layer(h, x, edge_index, d, flags, edge_mask)
        h = self.embedding_out(h) * flags
        
        if self.include_charge:
            charge = h[:, -1:]
            h = h[:, :self.nfeat-1]
        
        h = torch.nn.functional.softmax(h, dim=-1) * flags

        h = h.view(bs, n_nodes, -1)
        x = x.view(bs, n_nodes, -1) - x_
        if self.include_charge:
            charge = charge.view(bs, n_nodes, -1)
            h = torch.cat([h, charge], dim=-1)
        
        return h, x

    def make_edge_index(self, bs):
        edge_index = []
        for i in range(bs):
            edge_index.append(self.full_edge_index + (i * self.max_N))
        
        edge_index = torch.cat(edge_index, dim=1)

        return edge_index
