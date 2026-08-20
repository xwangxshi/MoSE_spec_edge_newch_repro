import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder or cfg.dataset.edge_encoder or (hasattr(cfg.dataset, "graph_encoder") and cfg.dataset.graph_encoder):
            if hasattr(cfg, "mlp"):
                dim_inner = cfg.mlp.dim_hidden
            elif hasattr(cfg, "gt"):
                if hasattr(cfg.gt, "varied_width") and cfg.gt.varied_width: #rez edit
                    dim_inner = cfg.gt.dim_initial
                else:
                    dim_inner = cfg.gt.dim_hidden
            elif hasattr(cfg, "gnn"):
                dim_inner = cfg.gnn.dim_inner
            else:
                raise Exception("Cannot find embedding dimension in cfg")
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, dim_inner)
            else:
                cfg.gnn.dim_edge = dim_inner
            #for GPS with varied layer width
            if hasattr(cfg, "gt") and hasattr(cfg.gt, "varied_width") and cfg.gt.varied_width:
                cfg.gnn.dim_edge = cfg.gt.dim_initial
            # Different edge embed dim when we run edge encoding in parallel to node encoding
            if cfg.model.type == "MLPeGraphReg":
                cfg.gnn.dim_edge = cfg.mlp.edge_encoder.dim_hidden

            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
        if hasattr(cfg.dataset, "graph_encoder") and cfg.dataset.graph_encoder:
            GraphEncoder = register.network_dict[cfg.dataset.graph_encoder_name]
            self.graph_encoder = GraphEncoder(dim_inner) #MLPGraphCountEncoder actually doesn't need any args, it takes everything straight from cfg
            # if cfg.dataset.graph_encoder_bn:
            #     self.graph_encoder_bn = BatchNorm1dNode(
            #         new_layer_config(???, -1, -1, has_act=False,
            #                          has_bias=False, cfg=cfg))


    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

        