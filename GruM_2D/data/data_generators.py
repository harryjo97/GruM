import torch
from torch.utils.data import random_split
import networkx as nx
import pickle
import os, sys

def preprocess(data_dir='data', dataset='sbm', measure_train_mmd=False):
    filename = f'{data_dir}/'
    if dataset == 'sbm':
        filename += 'sbm_200.pt'
    elif dataset == 'planar':
        filename += 'planar_64_200.pt'
    elif dataset == 'proteins':
        filename += 'proteins_100_500.pt'
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented.')

    if os.path.isfile(filename):
        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(filename)
        print(f'Dataset {filename} loaded from file')
        test_len = int(round(len(adjs)*0.2))
        train_len = int(round((len(adjs) - test_len)*0.8))
        val_len = len(adjs) - train_len - test_len

        train_set, val_set, test_set = random_split(adjs, [train_len, val_len, test_len], 
                                                    generator=torch.Generator().manual_seed(1234))
        train_graphs, val_graphs, test_graphs = tensor_to_graphs(train_set), tensor_to_graphs(val_set), \
                                                tensor_to_graphs(test_set)

        with open(f'{data_dir}/{dataset}.pkl', 'wb') as f:
            pickle.dump(obj=(train_graphs, val_graphs, test_graphs), file=f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if measure_train_mmd:
            sys.path.append(os.path.abspath(os.getcwd()))
            from evaluation.stats import degree_stats, orbit_stats_all, clustering_stats, spectral_stats, \
                                            eval_sbm, eval_planar
            kernel = 'tv'
            train_mmd_degree = degree_stats(test_graphs, train_graphs, kernel)
            train_mmd_4orbits = orbit_stats_all(test_graphs, train_graphs, kernel)
            train_mmd_clustering = clustering_stats(test_graphs, train_graphs, kernel)    
            train_mmd_spectral = spectral_stats(test_graphs, train_graphs, kernel)
            print(f'TV measures of Training set vs Test set: ')
            print(f'Deg.: {train_mmd_degree:.4f}, Clus.: {train_mmd_clustering:.4f} '
                    f'Orbits: {train_mmd_4orbits:.4f}, Spec.: {train_mmd_spectral:.4f}')
            if dataset=='sbm' or dataset=='planar':
                val_fn = eval_sbm if dataset=='sbm' else eval_planar
                train_uniq, train_uniq_non_iso, train_eval = val_fn(test_graphs, train_graphs)
                print(f'V.U.N.: {train_eval} | Uniq.: {train_uniq} | Uniq. & Nov.: {train_uniq_non_iso}')

def tensor_to_graphs(adjs):
    graph_list = [nx.from_numpy_array(adj.cpu().detach().numpy()) for adj in adjs]
    return graph_list


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sbm')
    parser.add_argument('--mmd', action='store_true')
    args = parser.parse_known_args()[0]

    preprocess(dataset=args.dataset, measure_train_mmd=args.mmd)