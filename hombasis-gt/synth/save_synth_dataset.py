import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp

import json
import numpy as np
import dill
import networkx as nx
from ogb.linkproppred import PygLinkPropPredDataset

from torch.utils.data import random_split
import torch_geometric.transforms as T
from torch_geometric.utils.convert import from_networkx

from pathlib import Path
import sys
import os

class CustomDataset(InMemoryDataset):
    def __init__(self,listOfDataObjects, root=None):
        super().__init__(root=root)
        self.data, self.slices = self.collate(listOfDataObjects)

def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 torch.datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])

    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]

def load_synth_dataset(filename, root, log10=False):
    bias_type='homcounts'
    dataset = dill.load(open(os.path.join(root, filename), 'rb'))
    
    pyg_dataset = []
    for graph in dataset:
        nxg = nx.from_graph6_bytes(graph['graph'])
        pyg = from_networkx(nxg)
        
        #initialize node features
        x_feat = torch.ones((pyg.num_nodes, 1))
        
        if bias_type == 'homcounts':
            graph_counts = []
            for v in graph['homcounts'].values():
                graph_counts.append(v)
            homcounts = torch.Tensor(graph_counts)

        if log10:
            homcounts = torch.nan_to_num(homcounts, neginf=-1.0)
            pyg_data = Data(
            x = x_feat,
            y = torch.Tensor([1.0]), #placeholder y value
            edge_index = pyg.edge_index,
            spectral_radius = torch.Tensor([np.real(graph['spectral_radius'])]),
            avg_clustering = torch.Tensor([graph['avg_clustering']]),
            fractional_domination = torch.Tensor([graph['fracdom' ]]),
            homcounts = homcounts,
            )
        else:
            pyg_data = Data(
                x = x_feat,
                y = torch.Tensor([1.0]), #placeholder y value
                edge_index = pyg.edge_index,
                spectral_radius = torch.Tensor([np.real(graph['spectral_radius'])]),
                avg_clustering = torch.Tensor([graph['avg_clustering']]),
                homcounts = homcounts,
            )
            
        # if bias_type == 'rwse':
        #     transform = T.AddRandomWalkPE(walk_length=20, attr_name=None)
        #     pyg_data = transform(pyg_data)

        pyg_dataset.append(pyg_data)
                
    generator = torch.Generator().manual_seed(42) # fixed seed for dataset split
    train_data, val_data, test_data = random_split(pyg_dataset, [.8, .1, .1], generator=generator)    
    
    print(train_data[0])
    
    feature_dim = pyg_dataset[0].homcounts.size()[1]
    print(feature_dim)
    
    return train_data, val_data, test_data, feature_dim

#takes in three torch.datasets and returns a list of three pyg InMemoryDatasets
def get_synth_datasets(tr_graphs, val_graphs, tst_graphs):
    tr_list = [tr_graphs[i] for i in range(len(tr_graphs))]
    val_list = [val_graphs[i] for i in range(len(val_graphs))]
    tst_list = [tst_graphs[i] for i in range(len(tst_graphs))]

    tr_graphs = CustomDataset(tr_list)
    val_graphs = CustomDataset(val_list)
    tst_graphs = CustomDataset(tst_list)
    return [tr_graphs, val_graphs, tst_graphs]

def save_synth_dataset_basic(homcounts,verbose=False):
    #takes in a choice of homcounts: 'spasm' or 'all5' and saves the appropriate (joined) dataset (includes blank node features and both possible targets) to main dataset directory
    #is not capble of loading multiple stacked counts. Use 'save_synth_dataset_stacked' for that

    raw_data_dir = os.path.join(Path(__file__).parent, "raw_data")

    if homcounts=='Spasm':
        filename='synth_c78.dill'
    elif homcounts=='All5':
        filename='synth_v5.dill'
    elif homcounts=='Spasm_log10':
        filename='synth_mixed_c78_log10.dill'
    elif homcounts=='All5_log10':
        filename='synth_mixed_v5_log10.dill'
    else:
        raise Exception('homcounts type not accepted')

    log10 = 'log10' in filename

    train, val, test, dim = load_synth_dataset(filename=filename, root=raw_data_dir, log10=log10)
    if verbose:
        print('XXXXXXXXXXXXX')
        print(f'train: {train}')
        print(f'train type: {type(train)}')
        print(f'val: {val}')
        print(f'val type: {type(val)}')
        print(f'test: {test}')
        print(f'test type: {type(test)}')
        print(f'train[0]: {train[0]}')
        print(f'train[0] type: {type(train[0])}')
        print(f'dim: {dim}')
        print('XXXXXXXXXXXXX')

    data_list = get_synth_datasets(train, val, test)
    if verbose:
        print(f'data_list[0]: {data_list[0]}')
        print(f'data_list[1]: {data_list[1]}')
        print(f'data_list[2]: {data_list[2]}')

        print(f'data_list[0] item: {data_list[0][1000]}')
        print(f'data_list[0] item type: {type(data_list[0][1000])}')
        print('XXXXXXXXXXXXX')


    dataset = join_dataset_splits(data_list)
    if verbose:
        print(f'DATASET: {dataset}')
        print(f'DATASET type: {type(dataset)}')
        print(f'DATASET attributes: {dir(dataset)}')
        print(f'DATASET[0]: {dataset[0]}')
        print(f'DATASET[0] type: {type(dataset[0])}')
        print(f'Saving DATASET')

    main_dir = Path(__file__).parent.parent.parent
    save_in_dir=f"{main_dir}/datasets/SYNTH-{homcounts}/processed"

    print(f'Saving in: {save_in_dir}')
    if not os.path.exists(save_in_dir):
        os.makedirs(save_in_dir)
    torch.save(dataset, os.path.join(save_in_dir,"joined.pt"))


# save_synth_dataset_basic('Spasm')

# save_synth_dataset_basic('All5')

#for log10 counts (Fracdom)
save_synth_dataset_basic('Spasm_log10')

save_synth_dataset_basic('All5_log10')