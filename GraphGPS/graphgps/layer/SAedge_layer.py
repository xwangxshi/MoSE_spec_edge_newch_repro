import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class MultiHeadAttentionEdgeLayer(nn.Module):
    """Self attention with edge biasing"""

    def __init__(self, in_dim, out_dim, num_heads, n_edge_types, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Embedding(n_edge_types, out_dim * num_heads) #Finds new edge embedding at each layer, do not use with TypeDictEdge
        
        self.E_fake = nn.Embedding(1, out_dim * num_heads)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, batch, edge_src, edge_dest):
        src = batch.K_h[edge_src]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[edge_dest]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E_total)  # (num real edges) x num_heads x out_dim
        #scale and softmax
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[edge_src] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_dest, dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, edge_dest, dim=0, out=batch.Z, reduce='add')

    def forward(self, batch, fake_edge_index, fake_edge_attr, edge_src, edge_dest):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        E = self.E(batch.edge_attr)

        E_fake = self.E_fake(fake_edge_attr)
        
        #Combine real edges with fake edges
        E_total = torch.cat([E, E_fake], dim=0)

        V_h = self.V(batch.x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E_total = E_total.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch, edge_src, edge_dest)

        h_out = batch.wV / (batch.Z + 1e-6)

        return h_out


class SAEdgeLayer(nn.Module):
    """MultiHeadAttentionEdgeLayer with feed forward as regularizers"""

    def __init__(self, in_dim, out_dim, num_heads, n_edge_types, dropout=0.0,
                 layer_norm=False, batch_norm=True, down_proj=None,
                 residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.down_proj = down_proj
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = MultiHeadAttentionEdgeLayer(in_dim=in_dim,
                                                    out_dim=out_dim // num_heads,
                                                    num_heads=num_heads,
                                                    n_edge_types=n_edge_types,
                                                    use_bias=use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim) if self.down_proj == None else nn.LayerNorm(self.down_proj['dim_hidden']) #we apply norm after dim_down projection when we varied widths

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim) if self.down_proj == None else nn.BatchNorm1d(self.down_proj['dim_hidden']) #we apply norm after dim_down projection when we varied widths

        # Project hidden dimension down when using varied width
        if self.down_proj != None:
            self.ff_down = nn.Linear(self.down_proj['dim_initial'], self.down_proj['dim_hidden'])

    def forward(self, batch, fake_edge_index, fake_edge_attr, edge_src, edge_dest):
        h = batch.x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch, fake_edge_index, fake_edge_attr, edge_src, edge_dest)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.down_proj != None: #varied width
            h = self.ff_down(h)

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)
