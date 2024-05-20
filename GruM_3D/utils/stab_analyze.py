### Code adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules
import torch
from utils import bond_order
import numpy as np
from torch.distributions.categorical import Categorical
from rdkit import Chem
import math
import networkx as nx

def check_valency(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    edges = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if 'QM9_3D' == dataset_info['name']:
                order = bond_order.get_bond_order(atom1, atom2, dist)
            elif 'GEOM_DRUGS' == dataset_info['name']:
                order = bond_order.geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order

            if order > 0:
                edges.append([i, j])

    nr_stable_bonds = 0
    stable_atoms = []
    unstable_atoms = []
    for i, (atom_type_i, nr_bonds_i) in enumerate(zip(atom_type, nr_bonds)):
        possible_bonds = bond_order.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)
        if is_stable: stable_atoms.append(i)
        else: unstable_atoms.append(i)

    molecule_stable = nr_stable_bonds == len(x)

    if len(edges) > 0:
        G = nx.Graph()
        G.add_edges_from(edges)
        connected = nx.is_connected(G)
    else: connected = 0

    return molecule_stable, nr_stable_bonds, len(x), edges, (stable_atoms, unstable_atoms), connected

def analyze_valency(h, x, flags, dataset_info):
    one_hot = h
    x = x
    node_mask = flags

    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    connected = 0

    processed_list = []
    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))

    mol_stable_list, mol_unstable_list = [], []
    atom_stable_list, atom_unstable_list = [], [] # For unstable mols, save the index of unstable atoms
    edges = []
    for i, mol in enumerate(processed_list):
        pos, atom_type = mol
        validity_results = check_valency(pos, atom_type, dataset_info)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

        if validity_results[0]: # if mol is stable
            mol_stable_list.append(i)
        else:
            mol_unstable_list.append(i)

        atom_stable, atom_unstable = validity_results[4]
        atom_stable_list.append(atom_stable), atom_unstable_list.append(atom_unstable)

        edges.append(validity_results[3])
        connected += int(validity_results[5])

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    fraction_connected_mol = connected / float(n_samples)
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'connected_mol': fraction_connected_mol
    }

    if dataset_info['name'] == 'QM9_3D':
        smiles_list_path = './data/QM9_3D/processed/smiles.pt'
    elif dataset_info['name'] == 'GEOM_DRUGS':
        smiles_list_path = './data/GEOM_DRUGS/processed/geom_drugs_smiles.txt'
    metrics = BasicMolecularMetrics(dataset_info, smiles_list_path)
    rdkit_metrics = metrics.evaluate(processed_list)

    return validity_dict, rdkit_metrics


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(positions, atom_types, dataset_info):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    for _, atom in enumerate(X):
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        a.SetAtomMapNum(_)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    return mol


def build_xae_molecule(positions, atom_types, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'QM9_3D':
                order = bond_order.get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'GEOM_DRUGS':
                order = bond_order.geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]

class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, smiles_list_path):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_info = dataset_info
        self.smiles_list_path = smiles_list_path

        # Retrieve dataset smiles only for qm9 currently.
        if self.smiles_list_path.endswith('pt'):
            self.dataset_smiles_list = torch.load(self.smiles_list_path)
        else:
            self.dataset_smiles_list = []
            with open(self.smiles_list_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                self.dataset_smiles_list.append(line.replace('\n', ''))


    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []

        for graph in generated:
            mol = build_molecule(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)

        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity = self.compute_validity(generated)
        
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        
        return {"validity": validity, "uniqueness": uniqueness, "novelty":novelty, "unique":unique}

class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        self.m = Categorical(torch.tensor(prob))

    def init_flags(self, max_node_num, n_samples=1):
        idx = self.m.sample((n_samples,))
        node_num = self.n_nodes[idx]

        flags = []

        for n in node_num:
            flag = torch.zeros((1, max_node_num), dtype=torch.float)
            flag[:, :n] = 1
            flags.append(flag)

        return torch.cat(flags, dim=0)
        