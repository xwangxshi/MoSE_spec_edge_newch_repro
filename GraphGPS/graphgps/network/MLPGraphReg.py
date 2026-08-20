import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
import sys
import torch.nn as nn
from torch_geometric.nn.models import MLP
from torch_geometric.utils import scatter


from pathlib import Path
import sys
import os
sys.path.append(os.path.join(Path(__file__).parent.parent.parent.parent, 'hombasis-gt','hombasis-bench', 'models'))
from layers import MLPReadout, MLPLayer
from typing import Optional
from graphgps.network.feature_encoder import FeatureEncoder

@register_network('MLPGraphReg')
class MLPGraphReg(nn.Module):
    def __init__(self, dim_in, dim_out, use_decoder=True, feature_encoder_in_dim=28):
        
        super(MLPGraphReg, self).__init__()
        self.hidden_dim = cfg.mlp.dim_hidden
        #self.count_dim: int,
        self.num_layers = cfg.mlp.layers
        self.batch_norm = cfg.mlp.batch_norm
        self.residual = cfg.mlp.residual_connection
        self.dropout = cfg.mlp.dropout
        self.readout = cfg.model.graph_pooling
        self.use_decoder = use_decoder #set to False when you are using as a submodule in an MLP with edge features
        self.feature_encoder_in_dim = feature_encoder_in_dim #set to 4 when you are using as a submodule in an MLP with edge features
        self.stack_on_h = False
        self.concate_dim = self.hidden_dim
        posenc = "posenc_RWSE" if hasattr(cfg,"posenc_RWSE") else None #edit to accomdate arbitrary posenc?
        if hasattr(cfg,"ctenc_NodeCountEnc") and hasattr(cfg.ctenc_NodeCountEnc, "stack_on_h"):
            self.stack_on_h = cfg.ctenc_NodeCountEnc.stack_on_h
            self.concate_dim += cfg.ctenc_NodeCountEnc.dim_ct
        if posenc != None and hasattr(cfg,f"{posenc}") and hasattr(getattr(cfg,f"{posenc}"), "stack_on_h"):
            self.stack_on_h = getattr(getattr(cfg,f"{posenc}"),"stack_on_h") or self.stack_on_h
            self.concate_dim += getattr(getattr(cfg,f"{posenc}"), "dim_pe")

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        
        # MLP layers
        if self.stack_on_h:
            self.convs = nn.ModuleList([MLPLayer(self.hidden_dim, self.hidden_dim, batch_norm=self.batch_norm, residual=self.residual, dropout=self.dropout) for _ in range(self.num_layers-1)])
            self.convs.insert(0, MLPLayer(self.concate_dim, self.hidden_dim, batch_norm=self.batch_norm, residual=self.residual, dropout=self.dropout))
        else:
            self.convs = nn.ModuleList([MLPLayer(self.hidden_dim, self.hidden_dim, batch_norm=self.batch_norm, residual=self.residual, dropout=self.dropout) for _ in range(self.num_layers)])

        # decoder
        Head = register.head_dict[cfg.mlp.head]
        self.post_mp = Head(dim_in=self.hidden_dim, dim_out=dim_out)
        
    def forward(self, batch):        
        
        batch = self.encoder(batch)
        h = batch.x
        for layer in self.convs:
            h = layer(x=h)
        batch.x = h
        batch = self.post_mp(batch)

        return batch

@register_network('MLPeGraphReg')
class MLPeGraphReg(nn.Module):
    def __init__(self, dim_in, dim_out, use_decoder=True, feature_encoder_in_dim=28):
        
        super(MLPeGraphReg, self).__init__()
        self.hidden_dim = cfg.mlp.dim_hidden
        #self.count_dim: int,
        self.num_layers = cfg.mlp.layers
        self.batch_norm = cfg.mlp.batch_norm
        self.residual = cfg.mlp.residual_connection
        self.dropout = cfg.mlp.dropout
        self.readout = cfg.model.graph_pooling
        self.use_decoder = use_decoder #set to False when you are using as a submodule in an MLP with edge features
        self.feature_encoder_in_dim = feature_encoder_in_dim #set to 4 when you are using as a submodule in an MLP with edge features
        self.stack_on_h = False
        self.concate_dim = self.hidden_dim

        self.E_num_layers = cfg.mlp.edge_encoder.layers
        self.E_hidden_dim = cfg.mlp.edge_encoder.dim_hidden
        self.E_residual = cfg.mlp.edge_encoder.residual_connection if hasattr(cfg.mlp.edge_encoder, "residual_connection") else self.residual
        self.E_batch_norm = cfg.mlp.edge_encoder.batch_norm if hasattr(cfg.mlp.edge_encoder, "batch_norm") else self.batch_norm
        self.E_dropout = cfg.mlp.edge_encoder.dropout if hasattr(cfg.mlp.edge_encoder, "dropout") else self.dropout

        self.graph_encoder = False
        if hasattr(cfg.mlp, "graph_encoder") and cfg.mlp.graph_encoder.layers>0:
            self.graph_encoder = True
            self.G_num_layers = cfg.mlp.graph_encoder.layers
            self.G_hidden_dim = cfg.mlp.graph_encoder.dim_hidden
            self.G_residual = cfg.mlp.graph_encoder.residual_connection if hasattr(cfg.mlp.graph_encoder, "residual_connection") else self.residual
            self.G_batch_norm = cfg.mlp.graph_encoder.batch_norm if hasattr(cfg.mlp.graph_encoder, "batch_norm") else self.batch_norm
            self.G_dropout = cfg.mlp.graph_encoder.dropout if hasattr(cfg.mlp.graph_encoder, "dropout") else self.dropout

        posenc = "posenc_RWSE" if hasattr(cfg,"posenc_RWSE") else None #edit to accomdate arbitrary posenc?
        if hasattr(cfg,"ctenc_NodeCountEnc") and hasattr(cfg.ctenc_NodeCountEnc, "stack_on_h"):
            self.stack_on_h = cfg.ctenc_NodeCountEnc.stack_on_h
            self.concate_dim += cfg.ctenc_NodeCountEnc.dim_ct
        if posenc != None and hasattr(cfg,f"{posenc}") and hasattr(getattr(cfg,f"{posenc}"), "stack_on_h"):
            self.stack_on_h = getattr(getattr(cfg,f"{posenc}"),"stack_on_h") or self.stack_on_h
            self.concate_dim += getattr(getattr(cfg,f"{posenc}"), "dim_pe")

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        
        # MLP layers
        if self.stack_on_h:
            self.convs = nn.ModuleList([MLPLayer(self.hidden_dim, self.hidden_dim, batch_norm=self.batch_norm, residual=self.residual, dropout=self.dropout) for _ in range(self.num_layers-1)])
            self.convs.insert(0, MLPLayer(self.concate_dim, self.hidden_dim, batch_norm=self.batch_norm, residual=self.residual, dropout=self.dropout))
        else:
            self.convs = nn.ModuleList([MLPLayer(self.hidden_dim, self.hidden_dim, batch_norm=self.batch_norm, residual=self.residual, dropout=self.dropout) for _ in range(self.num_layers)])

        # MLP layers for edge features:
        self.E_convs = nn.ModuleList([MLPLayer(self.E_hidden_dim, self.E_hidden_dim, batch_norm=self.E_batch_norm, residual=self.E_residual, dropout=self.E_dropout) for _ in range(self.E_num_layers)])

        if self.graph_encoder:
            self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
            self.G_convs = nn.ModuleList([MLPLayer(self.G_hidden_dim, self.G_hidden_dim, batch_norm=self.G_batch_norm, residual=self.G_residual, dropout=self.G_dropout) for _ in range(self.G_num_layers-1)])
            self.G_convs.insert(0, MLPLayer(self.hidden_dim+self.E_hidden_dim, self.G_hidden_dim, batch_norm=self.G_batch_norm, residual=self.G_residual, dropout=self.G_dropout))

        # decoder
        dec_dim_in=self.G_hidden_dim if self.graph_encoder else self.hidden_dim+self.E_hidden_dim
        dec_concat_e = not self.graph_encoder
        if hasattr(cfg.mlp.head, "type") and hasattr(cfg.mlp.head, "hidden_layers"):
            Head = register.head_dict[cfg.mlp.head.type]
            self.post_mp = Head(dim_in=dec_dim_in, dim_out=dim_out, L=cfg.mlp.head.hidden_layers, concat_e=dec_concat_e, pre_pooled=self.graph_encoder)
        else:
            Head = register.head_dict[cfg.mlp.head]
            self.post_mp = Head(dim_in=dec_dim_in, dim_out=dim_out,concat_e=dec_concat_e,pre_pooled=self.graph_encoder)

    def forward(self, batch):        
        
        batch = self.encoder(batch)
        h = batch.x
        e = batch.edge_attr

        L = len(self.convs)
        Le = len(self.E_convs)

        min_len = min(L, Le)
        diff_len = max(L, Le) - min_len

        for i in range(min_len):
            h = self.convs[i](x=h)
            e = self.E_convs[i](x=e)
        if diff_len != 0:
            for j in range(diff_len):
                if Le == min_len:
                    h = self.convs[min_len + j](x=h)
                else:
                    e = self.E_convs[min_len + j](x=e)
        
        if self.graph_encoder:
            G_h = self.pooling_fun(batch.x, batch.batch)
            G_e = self.pooling_fun(batch.edge_attr, batch.edge_attr_batch)
            G = torch.cat((G_h,G_e),dim=1)
            for layer in self.G_convs:
                G = layer(x=G)
            batch.graph_feature = G

        batch.x = h
        batch.edge_attr = e
        batch = self.post_mp(batch)

        return batch