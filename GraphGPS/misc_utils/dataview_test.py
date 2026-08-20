import torch
#import torch_geoemetric
import os
from torch_geometric.data import DataLoader

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

import torch_geometric.transforms as T

import graphgps

from graphgps.utils import custom_create_loader

from pathlib import Path
import sys
import os
sys.path.append(os.path.join(Path(__file__).parent.parent, 'hombasis-gt', 'qm9','data_GraphGym_QM9'))
from CustomDataset import CustomDataset

# sys.path.append('/data/coml-graphml/kell7068/thesis_code_1/hombasis-gt/qm9/data_GraphGym_QM9')
# from CustomDataset import CustomDataset

def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)



args = parse_args()

set_cfg(cfg)
cfg.set_new_allowed(True)
load_cfg(cfg, args)

custom_set_out_dir(cfg, 'test1', 'test2')
dump_cfg(cfg)

# datamodule = GraphGymDataModule()
# loader = datamodule.train_dataloader()
# loaders = create_loader()
# loader = loaders[0]
loaders = custom_create_loader()
loader = loaders[0]
for batch in loader:
    batch=batch

setattr(cfg, "accelerator", "cpu")
#model = create_model()

print(batch)
#print(f'batch.batch: {batch.batch}')

#for qm9
# print(f'batch.y: {batch.y}')
# print(f'batch.x:{batch.x}')
# print(f'batch.x size: {batch.x.size()}')
# print(f'first row of x: {batch.x[0]}')
# print(f'edge_attr: {batch.edge_attr}')
# print(f'edge_attr size: {batch.edge_attr.size()}')

#Debugging batched node+edge feature stack (for SA edge)
# print(f'batch.batch size {batch.batch.size()}')
# print(f'edge batch: {batch.edge_attr_batch}')
# print(f'edge batch size: {batch.edge_attr_batch.size()}')
# print(f'size of homcounts: {batch.graph_hom.size()}')

#for tree homs
# print(f'batch.counts: {batch.counts}')
# print(f'batch.counts size: {batch.counts.size()}')

#for zinc spasm
# print(f'batch.counts: {batch.counts}')
# print(f'batch.counts[0]: {batch.counts[0]}')
# print(f'batch.counts[2]: {batch.counts[2]}')
# print(f'batch.counts[10]: {batch.counts[10]}')

#for zinc tree homs with spasm
print(f'batch: {batch}')
print(f'batch.counts_spasm: {batch.counts_spasm}')
print(f'batch.counts_spasm size: {batch.counts_spasm.size()}')
print(f'batch.counts_wl: {batch.counts_wl}')
print(f'batch.counts_wl size: {batch.counts_wl.size()}')

#uncomment if need param count as well
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total trainable parameters:", total_params)
