from time import time
import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import numpy as np
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.data.separate import separate
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Batch

from rdkit import Chem
import imageio

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
RDLogger.DisableLog('rdApp.*')

from utils import build_geom_dataset

import yaml
from easydict import EasyDict as edict

class QM9_3D(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 3d coordinates.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.
    """

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root: str, 
                 scale_atomtype: float,
                 scale_pos: float,
                 scale_charge: float,
                 dataset_info,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):

        self.max_N = dataset_info['max_N']
        self.types = dataset_info['types']
        self.charges = dataset_info['charges']
        self.atom_decoder = dataset_info['atom_decoder']
        self.n_nodes = dataset_info['n_nodes']

        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        self.bond_decoder = [None, BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
        
        self.scale_atomtype = scale_atomtype
        self.scale_pos = scale_pos
        self.scale_charge = scale_charge

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.smiles_list = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'data.pt', f'smiles.pt']

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False)

        data_list = []
        smiles_list = []
        skip_num = 0
        max_N = 0
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                skip_num += 1
                continue

            if mol is None:
                skip_num += 1
                continue

            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)

            N = mol.GetNumAtoms()
            max_N = max(N, max_N)

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            pos = pos - pos.mean(dim=0, keepdim=True)

            type_idx = []
            charge_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(self.types[atom.GetSymbol()])
                charge_idx.append([self.charges[atom.GetSymbol()]])            

            type_idx = torch.tensor(type_idx, dtype=torch.long)
            charge_idx = torch.tensor(charge_idx, dtype=torch.float)
            x = F.one_hot(type_idx, num_classes=len(self.types)).float()

            row, col, edge_type = [], [], []

            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [self.bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(self.bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]

            data = Data(positions=pos, charges=charge_idx, one_hot=x, 
                        edge_index=edge_index, edge_type=edge_type, 
                        N=torch.tensor([N], dtype=torch.long))


            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(smiles_list, self.processed_paths[1])

    def get(self, idx):
        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        N = data.N[0]
        # Batch-wise
        one_hot, node_mask = to_dense_batch(data.one_hot, max_num_nodes=self.max_N)
        positions, _ = to_dense_batch(data.positions, max_num_nodes=self.max_N)
        charges, _ = to_dense_batch(data.charges, max_num_nodes=self.max_N)
    
        flags = node_mask.float()

        # Edge Mask
        adj = torch.ones((N, N), dtype=torch.bool)
        adj[range(N), range(N)] = False
        edge_mask = torch.zeros((self.max_N, self.max_N), dtype=torch.bool)
        edge_mask[:N, :N] = adj
        edge_mask = edge_mask.view(1, self.max_N * self.max_N).float()

        # Scale
        one_hot = one_hot * self.scale_atomtype
        charges = charges * self.scale_charge
        positions = positions * self.scale_pos
        
        node_feat = torch.cat([one_hot, charges], dim=-1)

        final_data = Data(node_feat=node_feat,
                        positions=positions,
                        flags=flags,
                        edge_mask=edge_mask,
                        num_nodes=data.N)

        return final_data

class GEOM_DRUGS(Dataset):
    def __init__(self, data_list, scale_pos, scale_atomtype, dataset_info):

        lengths = [s.shape[0] for s in data_list]
        argsort = np.argsort(lengths)
        self.data_list = [data_list[i] for i in argsort]

        self.max_N = dataset_info['max_N']
        self.atomic_number_list = torch.tensor(dataset_info['charges'])[None, :]
        self.atom_decoder = dataset_info['atom_decoder']
        self.n_nodes = dataset_info['n_nodes']

        self.scale_atomtype = scale_atomtype
        self.scale_pos = scale_pos


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data_list[idx]
             
        N = data.shape[0]
        
        positions = torch.from_numpy(data[:, -3:]).float()
        atom_types = torch.from_numpy(data[:, 0].astype(int)[:, None])
        one_hot = (atom_types == self.atomic_number_list).float()

        final_data = Data(node_feat=one_hot, positions=positions)

        # Batch-wise
        keys = ['node_feat', 'positions']
        for key in keys:
            final_data[key], node_mask = to_dense_batch(final_data[key], max_num_nodes=self.max_N)
        
        flags = node_mask.float()

        # Edge Mask
        adj = torch.ones((N, N), dtype=torch.bool)
        adj[range(N), range(N)] = False
        edge_mask = torch.zeros((self.max_N, self.max_N), dtype=torch.bool)
        edge_mask[:N, :N] = adj
        edge_mask = edge_mask.view(1, self.max_N * self.max_N).float()

        # Scale
        final_data.node_feat *= self.scale_atomtype
        final_data.positions *= self.scale_pos
        final_data.flags = flags
        final_data.edge_mask = edge_mask
        final_data.num_nodes = torch.tensor([N], dtype=torch.long)

        return final_data

def collate_fn(data_list):
    batch = Batch()
    for key in data_list[0].keys:
        values = []
        for i, data in enumerate(data_list):
            value = data[key]
            values.append(value)
        batch[key] = torch.cat(values, dim=0)

    return batch
        
def dataloader(config):
    start_time = time()

    dataset_info = yaml.load(open('./config/dataset_info_3D.yaml', 'r'), Loader=yaml.FullLoader)
    dataset_info = dataset_info[config.data.data]
    
    if config.data.data == 'QM9_3D':
        dataset = QM9_3D('./data/QM9_3D/', 
                        scale_atomtype=config.data.scale.atomtype, 
                        scale_pos=config.data.scale.pos, 
                        scale_charge=config.data.scale.charge,
                        dataset_info=dataset_info)
    
        train_idx = torch.load('data/QM9_3D/train_idx.pt')
        valid_idx = torch.ones((len(dataset),), dtype=torch.bool)
        valid_idx[train_idx] = False
        train_dataset = dataset[train_idx]
        test_dataset = dataset[valid_idx]
        
    elif config.data.data == 'GEOM_DRUGS':
        data_file = './data/GEOM_DRUGS/processed/geom_drugs_30.npy'
        if not os.path.exists(data_file):
            build_geom_dataset.extract_conformers('data/GEOM_DRUGS/raw', 'drugs_crude.msgpack', 'data/GEOM_DRUGS/processed', False, 30)
        train_data, val_data, test_data = build_geom_dataset.load_split_data(data_file, perm_path='data/GEOM_DRUGS/raw/geom_permutation.npy')
        train_dataset = GEOM_DRUGS(train_data,
                                scale_atomtype=config.data.scale.atomtype, 
                                scale_pos=config.data.scale.pos,
                                dataset_info=dataset_info)
        test_dataset = GEOM_DRUGS(test_data,
                                scale_atomtype=config.data.scale.atomtype, 
                                scale_pos=config.data.scale.pos,
                                dataset_info=dataset_info)

    train_dataloader = DataLoader(train_dataset, 
                            batch_size=config.data.batch_size, 
                            shuffle=True,
                            num_workers=8,
                            collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, 
                            batch_size=config.data.batch_size, 
                            shuffle=False,
                            num_workers=8,
                            collate_fn=collate_fn)

    print(f'{time() - start_time:.2f} sec elapsed for data loading')
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    data_file = './data/GEOM_DRUGS/processed/geom_drugs_30.npy'
    if not os.path.exists(data_file):
        build_geom_dataset.extract_conformers('data/GEOM_DRUGS/raw', 'drugs_crude.msgpack', 'data/GEOM_DRUGS/processed', False, 30)
    train_data, val_data, test_data = build_geom_dataset.load_split_data(data_file, perm_path='data/GEOM_DRUGS/raw/geom_permutation.npy')
    train_dataset = GEOM_DRUGS(train_data,
                               scale_atomtype=1.,
                               scale_pos=4.,
                               scale_charge=0.1)
    test_dataset = GEOM_DRUGS(test_data,
                               scale_atomtype=1.,
                               scale_pos=4.,
                               scale_charge=0.1)