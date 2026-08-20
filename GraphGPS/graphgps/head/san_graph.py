import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
from torch_geometric.nn.pool.glob import global_add_pool

import torch


@register_head('san_graph')
class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2, concat_e=False, pre_pooled=False):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        c = None
        if hasattr(cfg, "mlp"):
            c = cfg.mlp.head
        elif hasattr(cfg, "gnn"):
            if hasattr(cfg.gnn, "head_misc"):
                c = cfg.gnn.head_misc
        self.use_graphHC = c.use_graphHC if hasattr(c, "use_graphHC") else False
        self.graphHC_dim = None
        if self.use_graphHC:
            if hasattr(cfg.dataset, 'graph_encoder') and cfg.dataset.graph_encoder:
                self.graphHC_dim = cfg.ctenc_GraphCountEnc.dim_ct
            elif hasattr(c, "graphHC_dim"):
                self.graphHC_dim = c.graphHC_dim
            else:
                raise Exception('No graph homcount dimension given ')
            dim_in = dim_in + self.graphHC_dim

        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]()
        self.concat_e = concat_e
        self.pre_pooled = pre_pooled

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        if self.pre_pooled:
            graph_emb = batch.graph_feature
        else:
            graph_emb = self.pooling_fun(batch.x, batch.batch)
        if self.concat_e:
            graph_emb_e = self.pooling_fun(batch.edge_attr, batch.edge_attr_batch)
            graph_emb = torch.cat((graph_emb,graph_emb_e),dim=1)
        if self.use_graphHC:
            assert hasattr(batch, "graph_hom")
            if tuple(batch.graph_hom.size())[0] == tuple(batch.y.size())[0]:#if graph_hom file is truly graph level (pre-pooled)
                homs = batch.graph_hom
            elif tuple(batch.graph_hom.size())[0] == tuple(batch.x.size())[0]:#if graph_hom file really has node level counts, we should pool to get graph level
                homs = global_add_pool(batch.graph_hom, batch.batch)
                batch.graph_hom = homs
                assert tuple(homs.size())[0] == tuple(batch.y.size())[0]
            graph_emb = torch.cat((graph_emb, homs), dim=1)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label

#OG san_graph
# @register_head('san_graph')
# class SANGraphHead(nn.Module):
#     """
#     SAN prediction head for graph prediction tasks.

#     Args:
#         dim_in (int): Input dimension.
#         dim_out (int): Output dimension. For binary prediction, dim_out=1.
#         L (int): Number of hidden layers.
#     """

#     def __init__(self, dim_in, dim_out, L=2):
#         super().__init__()
#         self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
#         list_FC_layers = [
#             nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
#             for l in range(L)]
#         list_FC_layers.append(
#             nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
#         self.FC_layers = nn.ModuleList(list_FC_layers)
#         self.L = L
#         self.activation = register.act_dict[cfg.gnn.act]()

#     def _apply_index(self, batch):
#         return batch.graph_feature, batch.y

#     def forward(self, batch):
#         graph_emb = self.pooling_fun(batch.x, batch.batch)
#         for l in range(self.L):
#             graph_emb = self.FC_layers[l](graph_emb)
#             graph_emb = self.activation(graph_emb)
#         graph_emb = self.FC_layers[self.L](graph_emb)
#         batch.graph_feature = graph_emb
#         pred, label = self._apply_index(batch)
#         return pred, label


#TODOOOO: either edit SANGraphHead or the deep version below to implement batch norm/rez conn/dropout for hidden layers of san_graph, and some number of constant layers.

# @register_head('san_graph_deep')
# class SANGraphHead(nn.Module):
#     """
#     SAN prediction head for graph prediction tasks.

#     Args:
#         dim_in (int): Input dimension.
#         dim_out (int): Output dimension. For binary prediction, dim_out=1.
#         L (int): Number of hidden layers.
#     """

#     def __init__(self, dim_in, dim_out, layers=2, concat_e=False, residual=True, batch_norm= ):
#         super().__init__()
#         self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
#         list_FC_layers = [
#             nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
#             for l in range(L)]
#         list_FC_layers.append(
#             nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
#         self.FC_layers = nn.ModuleList(list_FC_layers)
#         self.layers = layers
#         self.activation = register.act_dict[cfg.gnn.act]()
#         self.concat_e = concat_e

#     def _apply_index(self, batch):
#         return batch.graph_feature, batch.y

#     def forward(self, batch):
#         graph_emb = self.pooling_fun(batch.x, batch.batch)
#         if self.concat_e:
#             graph_emb_e = self.pooling_fun(batch.edge_attr, batch.edge_attr_batch)
#             graph_emb = torch.cat((graph_emb,graph_emb_e),dim=1)
#         for l in range(self.L):
#             graph_emb = self.FC_layers[l](graph_emb)
#             graph_emb = self.activation(graph_emb)
#         graph_emb = self.FC_layers[self.L](graph_emb)
#         batch.graph_feature = graph_emb
#         pred, label = self._apply_index(batch)
#         return pred, label