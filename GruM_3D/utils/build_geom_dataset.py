### Code adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules
import msgpack
import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
from torch_geometric.data import Data
import argparse


def extract_conformers(data_dir, data_file, save_dir, remove_h, conformations):
    print("Processing GEOM_DRUGS dataset")
    os.makedirs(save_dir, exist_ok=True)
    drugs_file = os.path.join(data_dir, data_file)
    save_file = f"geom_drugs_{'no_h_' if remove_h else ''}{conformations}"
    smiles_list_file = 'geom_drugs_smiles.txt'
    number_atoms_file = f"geom_drugs_n_{'no_h_' if remove_h else ''}{conformations}"

    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))

    all_smiles = []
    all_number_atoms = []
    dataset_conformers = []
    mol_id = 0
    for i, drugs_1k in enumerate(unpacker):
        print(f"Unpacking file {i}...")
        for smiles, all_info in drugs_1k.items():
            all_smiles.append(smiles)
            conformers = all_info['conformers']
            # Get the energy of each conformer. Keep only the lowest values
            all_energies = []
            for conformer in conformers:
                all_energies.append(conformer['totalenergy'])
            all_energies = np.array(all_energies)
            argsort = np.argsort(all_energies)
            lowest_energies = argsort[:conformations]
            for id in lowest_energies:
                conformer = conformers[id]
                coords = np.array(conformer['xyz']).astype(float)        # n x 4
                if remove_h:
                    mask = coords[:, 0] != 1.0
                    coords = coords[mask]
                n = coords.shape[0]
                all_number_atoms.append(n)
                mol_id_arr = mol_id * np.ones((n, 1), dtype=float)
                id_coords = np.hstack((mol_id_arr, coords))

                dataset_conformers.append(id_coords)
                mol_id += 1

    print("Total number of conformers saved", mol_id)
    all_number_atoms = np.array(all_number_atoms)
    dataset = np.vstack(dataset_conformers)

    print("Total number of atoms in the dataset", dataset.shape[0])
    print("Average number of atoms per molecule", dataset.shape[0] / mol_id)

    # Save conformations
    np.save(os.path.join(save_dir, save_file), dataset)
    # Save SMILES
    with open(os.path.join(save_dir, smiles_list_file), 'w') as f:
        for s in all_smiles:
            f.write(s)
            f.write('\n')

    # Save number of atoms per conformation
    np.save(os.path.join(save_dir, number_atoms_file), all_number_atoms)
    print("Dataset processed.")


def load_split_data(conformation_file, val_proportion=0.1, test_proportion=0.1,
                    filter_size=None, perm_path = None):
    from pathlib import Path

    path = Path(conformation_file)
    base_path = path.parent.absolute()

    # base_path = os.path.dirname(conformation_file)
    all_data = np.load(conformation_file)  # 2d array: num_atoms x 5

    mol_id = all_data[:, 0].astype(int)

    conformers = all_data[:, 1:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)
    print(len(data_list))  

    # Filter based on molecule size.
    if filter_size is not None:
        # Keep only molecules <= filter_size
        data_list = [molecule for molecule in data_list
                     if molecule.shape[0] <= filter_size]

        assert len(data_list) > 0, 'No molecules left after filter.'

    # CAREFUL! Only for first time run:
    if perm_path is None:
        perm = np.random.permutation(len(data_list)).astype('int32')
        print('Warning, currently taking a random permutation for '
            'train/val/test partitions, this needs to be fixed for'
            'reproducibility.')
        assert not os.path.exists(os.path.join(base_path, 'geom_permutation.npy'))
        np.save(os.path.join(perm_path), perm)
        del perm

    perm = np.load(os.path.join(perm_path))
    
    data_list = [data_list[i] for i in perm]

    num_mol = len(data_list)
    val_index = int(num_mol * val_proportion)
    test_index = val_index + int(num_mol * test_proportion)
    val_data = data_list[:val_index]
    test_data = data_list[val_index:test_index]
    train_data = data_list[test_index:]
    # val_data, test_data, train_data = np.split(data_list, [val_index, test_index])
    return train_data, val_data, test_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformations", type=int, default=30,
                        help="Max number of conformations kept for each molecule.")
    parser.add_argument("--remove_h", action='store_true', help="Remove hydrogens from the dataset.")
    parser.add_argument("--data_dir", type=str, default='./data/GEOM_DRUGS/raw/')
    parser.add_argument("--data_file", type=str, default="drugs_crude.msgpack")
    parser.add_argument('--save_dir', type=str, default='./data/GEOM_DRUGS/processed/')
    args = parser.parse_args()
    # extract_conformers(args.data_dir, args.data_file, args.save_dir, args.remove_h, args.conformations)
    train_data, valid_data, test_data = load_split_data(os.path.join(args.save_dir,'geom_drugs_30.npy'), perm_path = os.path.join(args.data_dir, 'geom_permutation.npy'))
    print(train_data[:3])
    # train_dataset = GeomDrugsDataset(train_data)
    # for data in train_dataset:
    #     print(data)
    #     break
