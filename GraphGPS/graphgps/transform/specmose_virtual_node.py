"""Virtual-node transform for graphs with precomputed SpecMoSE edges."""

import torch
from torch_geometric.transforms import BaseTransform, VirtualNode


class SpecMoSEVirtualNode(BaseTransform):
    """Add a virtual node after attaching original-graph SpecMoSE signals."""

    def __init__(self):
        self.virtual_node = VirtualNode()

    def forward(self, data):
        if not hasattr(data, 'edge_specmose'):
            raise ValueError('SpecMoSEVirtualNode requires edge_specmose')

        num_original_nodes = data.num_nodes
        num_original_edges = data.num_edges
        data.node_is_original = torch.ones(
            num_original_nodes,
            dtype=torch.bool,
            device=data.edge_index.device,
        )
        data.edge_is_original = torch.ones(
            num_original_edges,
            dtype=torch.bool,
            device=data.edge_index.device,
        )

        # PyG appends False to node_is_original and edge_is_original while
        # extending every other node/edge tensor with its standard zero rows.
        return self.virtual_node(data)
