import torch
import torch.nn as nn
import torch

class GCL(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()

        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
         
        self.att_mlp = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Sigmoid())

    def forward(self, h, edge_index, edge_attr, flags, edge_mask):
        row, col = edge_index

        # Message
        msg = torch.cat([h[row], h[col], edge_attr], dim=1)
        msg = self.msg_mlp(msg)

        att = self.att_mlp(msg)
        msg = (msg * att) * edge_mask

        # Aggregation
        agg = unsorted_segment_sum(msg, row, num_segments=h.size(0),
                                normalization_factor=1,
                                aggregation_method='sum')

        agg = torch.cat([h, agg], dim=1)
        h = h + self.node_mlp(agg)
        return h * flags

class E3CoordLayer(nn.Module):
    def __init__(self, hidden_dim, coords_range, edge_dim):
        super().__init__()
        self.tanh = True
        self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        torch.nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def forward(self, h, x, edge_index, edge_attr, coord_diff, flags, edge_mask):
        row, col = edge_index

        msg = torch.cat([h[row], h[col], edge_attr], dim=1)
        trans = coord_diff * torch.tanh(self.coord_mlp(msg)) * self.coords_range
        tanns = trans * edge_mask

        agg = unsorted_segment_sum(trans, row, num_segments=x.size(0),
                                normalization_factor=1,
                                aggregation_method='sum')
        x = x + agg
        return x * flags

class E3Block(nn.Module):
    def __init__(self, nhid, coords_range, n_layers=2):
        super().__init__()
        edge_dim = 2
        self.coords_range = coords_range
        self.gcl = nn.ModuleList([])
        for _ in range(n_layers):
            gcl = GCL(nhid, nhid, edge_dim)
            self.gcl.append(gcl)

        self.e3_coord_layer = E3CoordLayer(nhid, coords_range, edge_dim)

    def forward(self, h, x, edge_index, d, flags, edge_mask):
        d_, coord_diff = coord2diff(x, edge_index)
        edge_attr = torch.cat([d_, d], dim=1)
        for gcl in self.gcl:
            h = gcl(h, edge_index, edge_attr, flags, edge_mask)
        x = self.e3_coord_layer(h, x, edge_index, edge_attr, coord_diff, flags, edge_mask)

        return h, x

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

def coord2diff(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, dim=1, keepdim=True)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + 1)
    return radial, coord_diff