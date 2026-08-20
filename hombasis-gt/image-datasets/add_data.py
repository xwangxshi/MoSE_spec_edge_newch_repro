import os
import json
import numpy as np

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.datasets import ZINC

from ogb.linkproppred import PygLinkPropPredDataset

def add_hom(hom_files, idx_list, root, dataset):       
    original_data = [dataset[i] for i in range(len(dataset))]
    
    all_hom_data = []
    if type(hom_files) == list:
        for hom_file in hom_files:
            hom_path = os.path.join(root, hom_file)
            hom_data = json.load(open(hom_path))
            all_hom_data.append(hom_data)
    elif type(hom_files) == str:
        hom_path = os.path.join(root, hom_files)
        hom_data = json.load(open(hom_path))
        all_hom_data.append(hom_data)
    else:
        raise Exception('Input a valid hom_file')
        
    homcount_dataset = []
    for graph_idx in range(len(original_data)):
        graph_counts = []
        for v_idx in range(len(original_data[graph_idx].x)):
            
            vertex_counts = []
            for hom_list in all_hom_data:
                raw_counts = hom_list[str(graph_idx)]['homcounts']

                # -- TEMP FIX FOR MISSING COUNTS? --
                missing_counter = 0
                if str(v_idx) not in raw_counts.keys():
                    counts_vec_size = len(raw_counts[list(raw_counts.keys())[0]])
                    raw_counts[str(v_idx)] = [0 for i in range(counts_vec_size)]
                    missing_counter+=1
                # ----------------------------------

                homcounts = raw_counts[str(v_idx)]
                vertex_counts += homcounts
                
            if len(idx_list) > 0:
                vertex_counts = np.array(vertex_counts)[idx_list].tolist()

            graph_counts.append(vertex_counts)
                                
        # Fix NAN values (mnist and cifar)
        counts = torch.nan_to_num(torch.Tensor(graph_counts), neginf=-1.0)
        
        homcount_dataset.append(
            Data(
                x = original_data[graph_idx].x, 
                edge_index = original_data[graph_idx].edge_index, 
                edge_attr = original_data[graph_idx].edge_attr, 
                y = original_data[graph_idx].y,
                counts = counts,
            )
        )
    
    print(f'number of missing node counts: {missing_counter}') #TEMP FIX FOR MISSING COUNTS?

    dataset.data, dataset.slices = dataset.collate(homcount_dataset)
    return dataset