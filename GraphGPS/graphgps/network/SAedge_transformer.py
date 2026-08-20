import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.models.gnn import FeatureEncoder
from graphgps.network.feature_encoder import FeatureEncoder
from torch_geometric.graphgym.register import register_network

from graphgps.layer.SAedge_layer import SAEdgeLayer

from graphgps.utils import negate_edge_index

@register_network('SAedge')
class SAEdgeTransformer(torch.nn.Module):
    """Transformer with edge embeddings biased self attention"""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        assert (cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in) or (hasattr(cfg.gt, "varied_width") and cfg.gt.varied_width), \
            "The inner and hidden dims must match."
        assert not cfg.dataset.edge_encoder, "Do not use TypeDictEdge with SAedge model (edge embeddings are re-computed in every layer of SAedge)"
        assert hasattr(cfg.gt, "edge_encoder_num_types"), "Do not use TypeDictEdge with SAedge model: provide edge_encoder_num_types in cfg.gt instead"

        if hasattr(cfg.gt, "varied_width") and cfg.gt.varied_width: #work around setting a default value for varied_width
            assert hasattr(cfg.gt, "initial_depth"), "provide number of layers with initial width"
            assert hasattr(cfg.gt, "dim_initial"), "provide initial layer depth"
        else:
            setattr(cfg.gt, "varied_width", False)

        layers = []
        for i in range(cfg.gt.layers):
            layers.append(SAEdgeLayer(in_dim=cfg.gt.dim_hidden if not (cfg.gt.varied_width and i<=cfg.gt.initial_depth-1) else cfg.gt.dim_initial,
                                    out_dim=cfg.gt.dim_hidden if not (cfg.gt.varied_width and i<=cfg.gt.initial_depth-1) else cfg.gt.dim_initial,
                                    num_heads=cfg.gt.n_heads,
                                    n_edge_types=cfg.gt.edge_encoder_num_types,
                                    dropout=cfg.gt.dropout,
                                    layer_norm=cfg.gt.layer_norm,
                                    batch_norm=cfg.gt.batch_norm,
                                    down_proj={'dim_initial': cfg.gt.dim_initial, "dim_hidden": cfg.gt.dim_hidden} if cfg.gt.varied_width and i==cfg.gt.initial_depth-1 else None,
                                    residual=cfg.gt.residual))
        self.trf_layers = torch.nn.ModuleList(layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):

        fake_edge_index = negate_edge_index(batch.edge_index, batch.batch)
        # fake edge types for non-existent edges
        # fake_edge_attr = torch.zeros(fake_edge_index.size(1), dtype=torch.int).to(torch.device('cpu')) #Use this job for interactive sessions
        fake_edge_attr = torch.zeros(fake_edge_index.size(1), dtype=torch.int).to(self.device) #Use this line when queueing batch jobs
        edge_index_total = torch.cat([batch.edge_index, fake_edge_index], dim=1) #CHANGE to individual columns
        edge_src = edge_index_total[0]
        edge_dest = edge_index_total[1]

        batch = self.encoder(batch)
        for layer in self.trf_layers:
            batch = layer(batch, fake_edge_index, fake_edge_attr, edge_src, edge_dest)
        batch = self.post_mp(batch)

        return batch
