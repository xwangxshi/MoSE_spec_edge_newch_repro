import os
import json
import numpy as np

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset


def add_peptide_hom(hom_files, idx_list, root, dataset, wl_count_label=False, preserve_counts_spasm=False):    
    original_data = [dataset[i] for i in range(len(dataset))]
    
    hom_path = os.path.join(root, hom_files)
    hom_data = json.load(open(hom_path))

    homcount_dim = len(hom_data['0']['homcounts']['0'])

    # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    # #print(hom_data[str(0)])
    # print(list(hom_data[str(0)].keys()))
    # print(hom_data[str(0)]['homcounts'])
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(hom_data[str(0)]['homcounts'][str(0)])
    print(list(hom_data[str(449)]['homcounts'].keys()))
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        
    homcount_dataset = []
    for graph_idx in range(len(original_data)):
        
        graph_counts = []
        for v_idx in range(len(original_data[graph_idx].x)):
            
            vertex_counts = []
            # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            # # print(list(hom_data[str(0)]['homcounts'].keys()))
            # print(f'graph_idx: {graph_idx}')
            # print(f'graph_idx: {v_idx}')
            # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            try:
                homcounts = hom_data[str(graph_idx)]['homcounts'][str(v_idx)]
            except:
                homcounts = [0]*homcount_dim
            # homcounts = hom_data[str(graph_idx)]['homcounts'][str(v_idx)]
            # homcounts = hom_data[str(graph_idx)]['homcounts'][str(v_idx)][:-2]
            vertex_counts += homcounts
        
                
            if len(idx_list) > 0:
                vertex_counts = np.array(vertex_counts)[idx_list].tolist()

            graph_counts.append(vertex_counts)

        if wl_count_label:
            if preserve_counts_spasm:
                homcount_dataset.append(
                    Data(
                        x = original_data[graph_idx].x, 
                        edge_index = original_data[graph_idx].edge_index, 
                        edge_attr = original_data[graph_idx].edge_attr, 
                        # edge_attr_ptr = original_data[graph_idx].edge_attr_ptr,
                        # ptr = original_data[graph_idx].ptr,
                        y = original_data[graph_idx].y,
                        counts_spasm = original_data[graph_idx].counts_spasm,
                        counts_wl = torch.Tensor(graph_counts),
                    )
                )
            else:
                homcount_dataset.append(
                    Data(
                        x = original_data[graph_idx].x, 
                        edge_index = original_data[graph_idx].edge_index, 
                        edge_attr = original_data[graph_idx].edge_attr, 
                        # edge_attr_ptr = original_data[graph_idx].edge_attr_ptr,
                        # ptr = original_data[graph_idx].ptr,
                        y = original_data[graph_idx].y,
                        counts_wl = torch.Tensor(graph_counts),
                    )
                )
        else:                                  
            homcount_dataset.append(
                Data(
                    x = original_data[graph_idx].x, 
                    edge_index = original_data[graph_idx].edge_index, 
                    edge_attr = original_data[graph_idx].edge_attr, 
                    # edge_attr_ptr = original_data[graph_idx].edge_attr_ptr,
                    # ptr = original_data[graph_idx].ptr,
                    y = original_data[graph_idx].y,
                    counts = torch.Tensor(graph_counts),
                )
            )

    dataset.data, dataset.slices = dataset.collate(homcount_dataset)
    return dataset