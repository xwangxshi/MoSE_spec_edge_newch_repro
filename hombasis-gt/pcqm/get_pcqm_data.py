import os
import json
import numpy as np

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset


def add_pcqm_hom(homcount_file, root, data_list):    
   
    hom_path = os.path.join(root, homcount_file)
    hom_data = json.load(open(hom_path))
    
    homcount_dim = len(hom_data['0']['homcounts']['0'])
    print(homcount_dim)
    
    homcount_dataset = []
    for graph_idx in range(len(data_list)):
        
        graph_counts = []
        for v_idx in range(len(data_list[graph_idx].x)):
            
            vertex_counts = []
            try:
                homcounts = hom_data[str(graph_idx)]['homcounts'][str(v_idx)]
            except:
                # print('in except')
                # print(graph_idx)
                # print(v_idx)
                homcounts = [0]*homcount_dim
            vertex_counts += homcounts

            graph_counts.append(vertex_counts)
                                
        homcount_dataset.append(
            Data(
                x = data_list[graph_idx].x, 
                edge_index = data_list[graph_idx].edge_index, 
                edge_attr = data_list[graph_idx].edge_attr, 
                y = data_list[graph_idx].y,
                counts = torch.Tensor(graph_counts),
            )
        )

    # dataset.data, dataset.slices = dataset.collate(homcount_dataset)
    return homcount_dataset