
import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gps_layer import GPSLayer
from graphgps.network.feature_encoder import FeatureEncoder


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in and (not hasattr(cfg.gt, "varied_width") or not cfg.gt.varied_width):
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        if hasattr(cfg.gt, "varied_width") and cfg.gt.varied_width: #work around setting a default value for varied_width
            assert hasattr(cfg.gt, "initial_depth"), "provide number of layers with initial width"
            assert hasattr(cfg.gt, "dim_initial"), "provide initial layer depth"
        else:
            setattr(cfg.gt, "varied_width", False)

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for i in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden if not (cfg.gt.varied_width and i<=cfg.gt.initial_depth-1) else cfg.gt.dim_initial,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
                down_proj={'dim_initial': cfg.gt.dim_initial, "dim_hidden": cfg.gt.dim_hidden} if cfg.gt.varied_width and i==cfg.gt.initial_depth-1 else None,
                initial_edge_dim=cfg.gt.dim_initial if cfg.gt.varied_width else None, #for down-projecting initial edge embeddings when using GINE with varied width
                n_edge_types = cfg.gnn.n_edge_types if hasattr(cfg.gnn, "n_edge_types") and not cfg.dataset.edge_encoder else None, #no need to embed edges types when using RGCN
                ff = cfg.gt.ff if hasattr(cfg.gt, "ff") else True,
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
