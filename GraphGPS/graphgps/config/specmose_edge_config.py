from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('specmose_edge')
def set_cfg_specmose_edge(cfg):
    """Configuration for the edge-restricted spectralized two-root MoSE."""
    cfg.specmose_edge = CN()
    cfg.specmose_edge.enable = False
    cfg.specmose_edge.cache_dir = (
        'hombasis-gt/hombasis-bench/data/zinc-data/'
        'zinc12k_two_root_mose_cheby_k10_edge_v1'
    )
    cfg.specmose_edge.num_templates = 42
    cfg.specmose_edge.num_basis = 21
    cfg.specmose_edge.max_total_degree = 5
    cfg.specmose_edge.num_learned_filters = 10
    cfg.specmose_edge.template_dim = 21
    cfg.specmose_edge.dim_struct = 42
    cfg.specmose_edge.dropout = 0.0
