import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP
from torch_geometric.nn.encoding import PositionalEncoding
from torch_geometric.utils import scatter
from models.layers import GCNLayer, GATLayer, GINLayer, MLPReadout, GINLayerSig, MLPLayer
from typing import Optional


class GINGraphReg(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            count_dim: int,
            # pe_dim: int,
            num_layers: int,
            batch_norm: bool,
            residual: bool,
            readout: str,
        ):
        
        super(GINGraphReg, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        # self.pe_dim = pe_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.readout = readout

        # 1-hot encode + linear node features
        self.atom_encoder = nn.Embedding(
            num_embeddings = 28, # num different atoms in ZINC
            embedding_dim = hidden_dim
        )
        
        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(in_channels=count_dim, hidden_channels=count_dim, out_channels=count_dim, num_layers=2)
            concat_feature_dim = hidden_dim + count_dim
        else:
            concat_feature_dim = hidden_dim
        
        # GIN message passing layers
        self.convs = nn.ModuleList([GINLayer(hidden_dim, hidden_dim, batch_norm=batch_norm, residual=residual) for _ in range(self.num_layers-1)])
        self.convs.insert(0, GINLayer(concat_feature_dim, hidden_dim, batch_norm=batch_norm, residual=residual))

        # decoder
        self.decoder = MLPReadout(hidden_dim, 1)

        
    def forward(self, x, edge_index, counts, use_counts, batch):        
        # encode features
        atom_h = self.atom_encoder(x)
        atom_h = torch.squeeze(atom_h)
        
        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((atom_h, count_h), dim=1)
            
        else:
            h = atom_h
            
        for layer in self.convs:
            h = layer(x=h, edge_index=edge_index)

        # decoding step
        h = scatter(h, batch, reduce=self.readout)
        h = self.decoder(h)
        
        return h

    
class GCNGraphReg(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            count_dim: int,
            num_layers: int,
            batch_norm: bool,
            residual: bool,
            readout: str,
        ):
        
        super(GCNGraphReg, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.readout = readout

        # 1-hot encode + linear node features
        self.atom_encoder = nn.Embedding(
            num_embeddings = 28, # num different atoms in ZINC
            embedding_dim = hidden_dim
        )
        
        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(in_channels=count_dim, hidden_channels=count_dim, out_channels=count_dim, num_layers=2)
        
        concat_feature_dim = hidden_dim + count_dim
        
        # GCN message passing layers        
        self.convs = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, batch_norm=batch_norm, residual=residual) for _ in range(self.num_layers-1)])
        self.convs.insert(0, GCNLayer(concat_feature_dim, hidden_dim, batch_norm=batch_norm, residual=residual))
        
        # decoder
        self.decoder = MLPReadout(hidden_dim, 1)
        
    def forward(self, x, edge_index, counts, use_counts, batch):        
        # encode features
        atom_h = self.atom_encoder(x)
        atom_h = torch.squeeze(atom_h)
        
        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((atom_h, count_h), dim=1)
        else:
            h = atom_h

        # model step
        for conv in self.convs:
            h = conv(x=h, edge_index=edge_index)

        # decoding step
        h = scatter(h, batch, reduce=self.readout)
        h = self.decoder(h)
        
        return h
    

class GATGraphReg(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            hidden_out_dim: int,
            count_dim: int,
            num_layers: int,
            num_heads:int,
            batch_norm: bool,
            residual: bool,
            readout: str,
        ):
        
        super(GATGraphReg, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        self.num_layers = num_layers
        self.readout = readout
        self.out_dim = hidden_out_dim

        head_hidden_dim = hidden_dim * num_heads
        # 1-hot encode + linear node features
        self.atom_encoder = nn.Embedding(
            num_embeddings = 28, # num different atoms in ZINC
            embedding_dim = head_hidden_dim
        )
        
        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(in_channels=count_dim, hidden_channels=count_dim, out_channels=count_dim, num_layers=2)
        
        concat_feature_dim = head_hidden_dim + count_dim
        
        self.prepare_gat = nn.Linear(concat_feature_dim, head_hidden_dim)
        
        # GAT message passing layers        
        self.convs = nn.ModuleList([GATLayer(head_hidden_dim, hidden_dim, num_heads, batch_norm=batch_norm, residual=residual) for _ in range(self.num_layers-1)])
        self.convs.append(GATLayer(head_hidden_dim, hidden_out_dim, num_heads=1, batch_norm=batch_norm, residual=residual))
        
        # decoder
        self.decoder = MLPReadout(hidden_out_dim, 1)
        
    def forward(self, x, edge_index, counts, use_counts, batch):        
        # encode features
        atom_h = self.atom_encoder(x)
        atom_h = torch.squeeze(atom_h)
        
        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((atom_h, count_h), dim=1)
            h = self.prepare_gat(h)
        else:
            h = atom_h

        # model step
        for conv in self.convs:
            h = conv(x=h, edge_index=edge_index)

        # decoding step
        h = scatter(h, batch, reduce=self.readout)
        h = self.decoder(h)
        
        return h
    

class MLPGraphReg(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            count_dim: int,
            num_layers: int,
            batch_norm: bool,
            residual: bool,
            dropout: int,
            readout: str,
            use_decoder: Optional[bool] = True, #set to False when you are using as a submodule in an MLP with edge features
            feature_encoder_in_dim: Optional[int] = 28, #set to 4 when you are using as a submodule in an MLP with edge features
            expand_x: Optional[bool] = False
        ):
        
        super(MLPGraphReg, self).__init__()
        self.hidden_dim = hidden_dim
        self.count_dim = count_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.readout = readout

        # 1-hot encode + linear node features
        embedding_dim = hidden_dim if expand_x == True else hidden_dim - count_dim
        self.atom_encoder = nn.Embedding(
            num_embeddings = feature_encoder_in_dim, # num different atoms in ZINC
            embedding_dim = embedding_dim
        )
        
        # encode homcounts in 2-layer MLP
        if count_dim > 0:
            self.count_encoder = MLP(in_channels=count_dim, hidden_channels=count_dim, out_channels=count_dim, num_layers=2)
            concat_feature_dim = hidden_dim + count_dim if expand_x == True else hidden_dim
        else:
            concat_feature_dim = hidden_dim
        
        # MLP layers
        self.convs = nn.ModuleList([MLPLayer(hidden_dim, hidden_dim, batch_norm=batch_norm, residual=residual, dropout=dropout) for _ in range(self.num_layers-1)])
        self.convs.insert(0, MLPLayer(concat_feature_dim, hidden_dim, batch_norm=batch_norm, residual=residual, dropout=dropout))

        # decoder
        self.decoder = MLPReadout(hidden_dim, 1) if use_decoder else None

        
    def forward(self, x, counts, use_counts, batch):        
        # encode features
        atom_h = self.atom_encoder(x)
        atom_h = torch.squeeze(atom_h)
        
        if use_counts:
            count_h = self.count_encoder(counts)
            h = torch.cat((atom_h, count_h), dim=1)
            
        else:
            h = atom_h
            
        for layer in self.convs:
            h = layer(x=h)

        # decoding step
        h = scatter(h, batch, reduce=self.readout)
        h = self.decoder(h) if self.decoder != None else h
        
        return h

class MLPedgeGraphReg(nn.Module):
    def __init__(
        self,
        hidden_dim_h: int,
        count_dim: int,
        num_layers_h: int,
        batch_norm_h: bool,
        residual_h: bool,
        dropout_h: int,
        readout_h: str,
        hidden_dim_e: Optional[int] = None ,
        num_layers_e: Optional[int] = None,
        batch_norm_e: Optional[bool] = None,
        residual_e: Optional[bool] = None,
        dropout_e: Optional[int] = None,
        readout_e: Optional[str] = None,
    ):
    
        super(MLPedgeGraphReg, self).__init__()
        self.hidden_dim_h = hidden_dim
        self.count_dim = count_dim
        self.num_layers_h = num_layers
        self.batch_norm_h = batch_norm
        self.readout_h = readout
        self.residual_h = residual_h
        self.dropout_h = dropout_h

        self.hidden_dim_e = hidden_dim_e if hidden_dim_e != None else hidden_dim_h
        self.num_layers_e = num_layers_e if num_layers_e != None else num_layers_h
        self.batch_norm_e = batch_norm_e if batch_norm_e != None else batch_norm_h
        self.readout_e = readout_e if readout_e != None else readout_h
        self.residual_e = residual_e if residual_e != None else residual_h
        self.dropout_e = dropout_e if dropout_e != None else dropout_h

        MLP_h = MLPGraphReg(
                hidden_dim=self.hidden_dim_h,
                count_dim=self.count_dim,
                num_layers=self.num_layers_h,
                batch_norm=self.batch_norm_h,
                residual=self.residual_h,
                dropout=self.dropout_h,
                readout=self.readout_h,
                use_decoder = False
            )
        MLP_e = MLPGraphReg(
                hidden_dim=self.hidden_dim_e,
                count_dim=0,
                num_layers=self.num_layer_e,
                batch_norm=self.batch_norm_e,
                residual=self.residual_e,
                dropout=self.dropout_e,
                readout=self.readout_e,
                use_decoder = False,
                feature_encoder_in_dim=4
            )
        self.decoder = MLPReadout(hidden_dim_h + hidden_dim_e, 1)

    def forward(self, x, edge_attr, counts, use_counts, batch, batch_e):

        h = MLP_h(x=x, counts=counts, use_counts=use_counts, batch=batch)
        e = MLP_e(x=edge_attr, counts=counts, use_counts=False, batch=batch_e)

        return self.decoder(torch.cat((h, e.unsqueeze(-1)), dim=1))

