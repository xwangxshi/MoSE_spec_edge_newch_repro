"""Edge encoder for degree-truncated spectralized two-root MoSE signals."""

import torch
from torch import nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_edge_encoder
from torch_geometric.nn import MLP


def signed_log1p(values):
    return torch.sign(values) * torch.log1p(torch.abs(values))


@register_edge_encoder('TypeDictEdge+SpecMoSE')
class TypeDictSpecMoSEEdgeEncoder(nn.Module):
    """Fuse bond types with learned filters of the SpecMoSE basis tensor."""

    def __init__(self, emb_dim):
        super().__init__()
        spec_cfg = cfg.specmose_edge
        if not spec_cfg.enable:
            raise ValueError('specmose_edge.enable must be True')
        if spec_cfg.dim_struct >= emb_dim:
            raise ValueError('SpecMoSE structural width must be below edge width')

        self.num_templates = spec_cfg.num_templates
        self.num_basis = spec_cfg.num_basis
        self.num_learned_filters = spec_cfg.num_learned_filters
        self.template_dim = spec_cfg.template_dim
        self.dim_struct = spec_cfg.dim_struct

        self.bond_encoder = nn.Embedding(
            cfg.dataset.edge_encoder_num_types,
            emb_dim - self.dim_struct,
        )
        self.filter_mixer = nn.Linear(
            self.num_basis,
            self.num_learned_filters,
            bias=False,
        )
        self.template_mixer = nn.Linear(
            self.num_templates,
            self.template_dim,
            bias=False,
        )

        mixed_dim = (1 + self.num_learned_filters) * self.template_dim
        self.structural_encoder = MLP(
            in_channels=mixed_dim,
            hidden_channels=self.dim_struct,
            out_channels=self.dim_struct,
            num_layers=2,
            dropout=spec_cfg.dropout,
            norm='batch_norm',
        )

    def forward(self, batch):
        responses = batch.edge_specmose
        if responses.shape[1:] != (self.num_templates, self.num_basis):
            raise ValueError(
                f'Expected edge_specmose[:, {self.num_templates}, '
                f'{self.num_basis}], got {tuple(responses.shape)}'
            )

        # Keep the raw invariant as a fixed structural filter and learn the
        # remaining filters as bias-free combinations of the selected basis.
        raw_response = responses[:, :, :1]
        learned_responses = self.filter_mixer(responses)
        filter_responses = torch.cat(
            [raw_response, learned_responses], dim=2
        )
        filter_responses = signed_log1p(filter_responses)

        # Apply the same template mixer independently to every filter slot.
        template_features = self.template_mixer(
            filter_responses.transpose(1, 2)
        )
        mixed_features = template_features.flatten(start_dim=1)
        structural_embedding = self.structural_encoder(mixed_features)

        # A genuinely zero structural signal must remain zero (also the
        # convention for future virtual-node edges).
        has_signal = responses.ne(0).any(dim=(1, 2), keepdim=False)
        structural_embedding = structural_embedding * has_signal[:, None]

        bond_embedding = self.bond_encoder(batch.edge_attr)
        batch.edge_attr = torch.cat(
            [bond_embedding, structural_embedding], dim=1
        )
        return batch
