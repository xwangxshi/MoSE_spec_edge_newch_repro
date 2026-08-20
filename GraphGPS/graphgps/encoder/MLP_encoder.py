import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.nn.models import MLP


@register_node_encoder('MLPNodeEnc')
class MLPNodeEncoder(torch.nn.Module):
    def __init__(self, dim_emb):
        super().__init__()

        self.dim_in = cfg.dataset.node_encoder_init_dim
        self.dim_hid = cfg.dataset.node_encoder_hid_dim
        self.L = cfg.dataset.node_encoder_depths
        self.bn = "batch_norm" if cfg.dataset.node_encoder_bn else None
    
        self.encoder = MLP(in_channels=self.dim_in, hidden_channels=self.dim_hid, out_channels=dim_emb, num_layers=self.L, norm=self.bn)

    def forward(self, batch):
    
        batch.x = self.encoder(batch.x)
        return batch