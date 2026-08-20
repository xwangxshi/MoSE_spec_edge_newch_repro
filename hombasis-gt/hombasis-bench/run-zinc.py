import os
import numpy as np
import random

import torch
from torch.optim import Adam, AdamW
import torch_geometric.seed
from torch_geometric.loader import DataLoader
import wandb
from torch.utils.data import Subset

from data.get_data import load_zinc_homcount_dataset, load_zinc_subcount_dataset, load_zinc_subhom_dataset
from models.graph_reg import GINGraphReg, GCNGraphReg, GATGraphReg, MLPGraphReg
from zinc_utils import train, eval, parse_args

if __name__ == "__main__":
    # set device
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

    # get args
    conf = parse_args()
    print(conf)

    wandb.init(
        project=conf['project'],
        group=conf['group'],
        name=conf['model']
    )

    wandb.config.update(conf)
    
    default_seed = int(conf['seed'])
    print("==========================================================")
    print("Using device", str(device))
    print("Seed:", str(default_seed))
    print("======================== Args ===========================")
    print(conf)
    print("===================================================")

    # Set the seed for everything
    torch.manual_seed(default_seed)
    torch.cuda.manual_seed(default_seed)
    torch.cuda.manual_seed_all(default_seed)
    np.random.seed(default_seed)
    random.seed(default_seed)
    torch_geometric.seed.seed_everything(default_seed)
    
    print("Loading Data")
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'data', 'zinc-data')
    model_name = conf['model']
    batch_size = int(conf['batch_size'])
    use_counts = conf['use_counts']
    count_files = conf['count_files']
    count_type = conf['count_type']
    idx_list = conf['idx_list']
    hidden_dim = int(conf['hidden_dim'])
    description = conf['description']
    n_epochs = conf['epochs']
    dropout = conf['dropout'] if 'dropout' in conf else 'none'

    
    if count_type == "homcounts" or count_type == "none":
        print('loading homcounts')
        train_data, val_data, test_data, count_dim = load_zinc_homcount_dataset(name='ZINC', hom_files=count_files, idx_list=idx_list, root=data_dir)
    elif count_type == "subcounts":
        sub_file = count_files[0]
        print('loading subcounts')
        train_data, val_data, test_data, count_dim = load_zinc_subcount_dataset(name='ZINC', sub_file=sub_file, idx_list=idx_list, root=data_dir)        
    elif count_type == "both":
        print('loading hom and subcounts')
        if "anchor" in count_files[0]:
            sub_file = 'zinc_3to10C_subgraph.json'
        else:
            sub_file = 'zinc_3to8C_multhom.json'
        train_data, val_data, test_data, count_dim = load_zinc_subhom_dataset(name='ZINC', hom_files=count_files, idx_list=idx_list, sub_file=sub_file, root=data_dir)
        
    else:
        print('count type not supported')
    
    if not use_counts:
        count_files = 'none'
        count_dim = 0
        
    print(count_dim)

    if 'debug' in conf and conf["debug"] == True:
        train_data = Subset(train_data, [i for i in range(100)])
        test_data = Subset(test_data, [i for i in range(100)])
        test_data = Subset(test_data, [i for i in range(100)])
        n_epochs = 5
        batch_size = 10
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, follow_batch=["edge_attr"])
    valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, follow_batch=["edge_attr"])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, follow_batch=["edge_attr"])


    print("Done Loading Data")

    print("Preparing Model")

    if model_name == "GINGraphReg":
        model = GINGraphReg(
            hidden_dim=hidden_dim,
            count_dim=count_dim,
            num_layers=int(conf['num_layers']),
            batch_norm=conf['batch_norm'],
            residual=conf['residual'],
            readout=conf['readout']
        )
    elif model_name == "GCNGraphReg":
        model = GCNGraphReg(
            hidden_dim=hidden_dim,
            count_dim=count_dim,
            num_layers=int(conf['num_layers']),
            batch_norm=conf['batch_norm'],
            residual=conf['residual'],
            readout=conf['readout']
        )
    elif model_name == "GATGraphReg":
        model = GATGraphReg(
            hidden_dim=hidden_dim,
            hidden_out_dim=conf['hidden_out_dim'],
            count_dim=count_dim,
            num_layers=int(conf['num_layers']),
            num_heads=conf['num_heads'],
            batch_norm=conf['batch_norm'],
            residual=conf['residual'],
            readout=conf['readout']
        )
    elif model_name == "MLPGraphReg":
        model = MLPGraphReg(
            hidden_dim=hidden_dim,
            count_dim=count_dim,
            num_layers=int(conf['num_layers']),
            batch_norm=conf['batch_norm'],
            residual=conf['residual'],
            dropout=dropout,
            readout=conf['readout']
        )
    elif model_name == "MLPedgeGraphReg":
        hidden_dim_e = int(conf['hidden_dim_e']) if 'hidden_dim_e' in conf else hidden_dim
        num_layers_e = int(conf['num_layers_e']) if 'num_layers_e' in conf else int(conf['num_layers'])
        batch_norm_e = bool(conf['batch_norm_e']) if 'batch_norm_e' in conf else conf['batch_norm']
        residual_e = bool(conf['residual_e']) if 'residual_e' in conf else conf['residual']
        dropout_e = float(conf['dropout_e']) if 'dropout_e' in conf else dropout
        readout_e = str(conf['readout_h']) if 'readout_h' in conf else conf['readout']

        model = MLPedgeGraphReg(
            hidden_dim_h=hidden_dim,
            count_dim=count_dim,
            num_layers_h=int(conf['num_layers']),
            batch_norm_h=conf['batch_norm'],
            residual_h=conf['residual'],
            dropout_h=dropout,
            readout_h=conf['readout'],
            hidden_dim_e=hidden_dim_e,
            num_layers_e=num_layers_e,
            batch_norm_e=batch_norm_e,
            residual_e=residual_e,
            dropout_e=dropout_e,
            readout_e=readout_e
        )
    else:
        print('model not supported')
        
    model.to(device)

    # instantiate optimiser
    weight_decay = float(conf['weight_decay']) if "weight_decay" in conf else 0
    if "optimizer" not in conf or conf['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=float(conf['init_lr']), weight_decay=weight_decay)
    elif conf['optimizer'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=float(conf['init_lr']), weight_decay=weight_decay)

    # learning rate decay
    if 'scheduler' not in conf or conf['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=conf['lr_factor'], patience=conf['lr_patience'], verbose=True)
    elif conf['scheduler'] == "StepLR":
        step_size = int(conf['step_size']) if 'step_size' in conf else 10
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    elif conf['scheduler'] == "CosineAnnealingLR":
        eta_min = float(conf['eta_min']) if 'eta_min' in conf else .00005
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=eta_min, last_epoch=-1)
    if 'lr_warmup' in conf and bool(conf['lr_warmup']) == True:
        warmup_epochs = int(conf['warmup_epochs']) if 'warmup_epochs' in conf else 10
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.05, end_factor=1, total_iters=warmup_epochs
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_epochs+1],
            )

        

    # (!) start training/evaluation
    best_val_epoch = 0
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    params = []

    print("Starting Model Training")

    # training
    for epoch in range(n_epochs):
        epoch_train_curve = train(model_name, use_counts, model, device, train_loader, optimizer)
        train_loss_curve += epoch_train_curve
        epoch_train_loss = float(np.mean(epoch_train_curve))

        if epoch % 20 == 0:
            print("=====Epoch {}".format(epoch))
        
        train_perf, _ = eval(model_name, use_counts, model, device, train_loader)
        train_curve.append(train_perf)
        
        valid_perf, epoch_val_loss = eval(model_name, use_counts, model, device, valid_loader)
        valid_curve.append(valid_perf)

        test_perf, epoch_test_loss = eval(model_name, use_counts, model, device, test_loader)
        test_curve.append(test_perf)
        
        wandb.log(
            {
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "test_loss": epoch_test_loss,
                "train_score": train_perf,
                "val_score": valid_perf,
                "test_score": test_perf,
            }
        )
        
        # decay learning rate
        if 'scheduler' not in conf or conf['scheduler'] == 'ReduceLROnPlateau':
            if 'lr_warmup' in conf and bool(conf['lr_warmup']) == True and epoch <= warmup_epochs+1:
                scheduler.step()
            else:
                scheduler.step(valid_perf)
        else:
            scheduler.step()
        if optimizer.param_groups[0]['lr'] < conf['min_lr']:
            print("\n!! The minimum learning rate has been reached.")
            break

    best_val_epoch = np.argmin(np.array(valid_curve))
    
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    
    print('Final Evaluation...')
    final_train_perf = np.nan
    final_val_perf = np.nan
    final_test_perf = np.nan
    final_train_perf, _ = eval(model_name, use_counts, model, device, train_loader)
    final_val_perf, _ = eval(model_name, use_counts, model, device, valid_loader)
    final_test_perf, _ = eval(model_name, use_counts,  model, device, test_loader)
    
    # save results
    wandb.log({
        'last_val': final_val_perf,
        'last_test': final_test_perf,
        'last_train': final_train_perf,
        'best_epoch': best_val_epoch,
        'best_train': train_curve[best_val_epoch],
        'best_val': valid_curve[best_val_epoch],
        'best_test': test_curve[best_val_epoch],
        'hidden_dim': hidden_dim,
        'use_counts': use_counts,
        'count_type': count_type,
        'count_files': count_files,
        'count_dim': count_dim,
        'description': description,
        'final_epoch': epoch,
        'total_params': total_params,
        "dropout": dropout
    })


    msg = (
       f'========== Result ============\n'
       f'Dataset:        ZINC\n'
       f'Countfile:        {count_files}\n'
       f'------------ Best epoch -----------\n'
       f'Train:          {train_curve[best_val_epoch]}\n'
       f'Validation:     {valid_curve[best_val_epoch]}\n'
       f'Test:           {test_curve[best_val_epoch]}\n'
       f'Best epoch:     {best_val_epoch}\n'
       '------------ Last epoch -----------\n'
       f'Train:          {final_train_perf}\n'
       f'Validation:     {final_val_perf}\n'
       f'Test:           {final_test_perf}\n'
       '-------------------------------\n\n')
    print(msg)

    wandb.finish()

