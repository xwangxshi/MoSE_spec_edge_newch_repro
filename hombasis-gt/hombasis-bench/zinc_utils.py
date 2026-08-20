import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error



def train(model_name, use_counts, model, device, loader, optimizer):
    """
        Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """

    loss_fn = torch.nn.L1Loss()

    curve = list()
    model.train()
    # for step, batch in enumerate(tqdm(loader, desc="Training iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)  # batch.cuda() if torch.cuda.is_available() else batch

        optimizer.zero_grad()
        
        if model_name == "GINGraphReg" or model_name == "GCNGraphReg" or model_name == "GATGraphReg":
            pred = model(x=batch.x, edge_index=batch.edge_index, counts=batch.counts, use_counts=use_counts, batch=batch.batch)
        elif model_name == "MLPGraphReg":
            pred = model(x=batch.x, counts=batch.counts, use_counts=use_counts, batch=batch.batch)
        elif model_name == "MLPedgeGraphReg":
            if hasattr(batch, edge_attr_batch):
                pred = model(x=batch.x, edge_attr=batch.edge_attr ,counts=batch.counts, use_counts=use_counts, batch=batch.batch, batch_e=batch.edge_attr_batch)
            else:
                print('Dataloader lacks edge_attr_batch tensor')
        else:
            print('model not supported')
            
        targets = batch.y.to(torch.float32).view(pred.shape)

        loss = loss_fn(pred, targets)

        loss.backward()
        optimizer.step()
        curve.append(loss.detach().cpu().item())

    return curve


def eval(model_name, use_counts, model, device, loader):
    """
        Evaluates a model over all the batches of a data loader.
    """
        
    loss_fn = torch.nn.L1Loss()

    model.eval()
    y_true = []
    y_pred = []
    losses = []
    # for step, batch in enumerate(tqdm(loader, desc="Eval iteration")):
    for step, batch in enumerate(loader):
        # Cast features to double precision if that is used
        if torch.get_default_dtype() == torch.float64:
            for dim in range(batch.dimension + 1):
                batch.cochains[dim].x = batch.cochains[dim].x.double()
                assert batch.cochains[dim].x.dtype == torch.float64, batch.cochains[dim].x.dtype

        batch = batch.to(device)
        with torch.no_grad():
            if model_name == "GINGraphReg" or model_name == "GCNGraphReg" or model_name == "GATGraphReg":
                pred = model(x=batch.x, edge_index=batch.edge_index, counts=batch.counts, use_counts=use_counts, batch=batch.batch)
            elif model_name == "MLPGraphReg":
                pred = model(x=batch.x, counts=batch.counts, use_counts=use_counts, batch=batch.batch)
            elif model_name == "MLPedgeGraphReg":
                if hasattr(batch, edge_attr_batch):
                    pred = model(x=batch.x, edge_attr=batch.edge_attr ,counts=batch.counts, use_counts=use_counts, batch=batch.batch, batch_e=batch.edge_attr_batch)
                else:
                    print('Dataloader lacks edge_attr_batch tensor')
            else:
                print('model not supported')
            
            targets = batch.y.to(torch.float32).view(pred.shape)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            loss = loss_fn(pred, targets)
            losses.append(loss.detach().cpu().item())

        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy() if len(y_true) > 0 else None
    y_pred = torch.cat(y_pred, dim=0).numpy()
    
    assert y_true is not None
    assert y_pred is not None
    
    mae = mean_absolute_error(y_true, y_pred)
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan
    
    return mae, mean_loss


def parse_args():
    parser = argparse.ArgumentParser(description='ZINC experiment')
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-project", "--project")
    parser.add_argument("-group","--group", help="group name on wandb", required=True)
    parser.add_argument("-seed", "--seed", help="seed", required=True)

    parser.add_argument("-num_layers", "--num_layers", help="Number of layers", required=False)
    parser.add_argument("-hidden_dim", "--hidden_dim", help="Dimension of hidden layers", required=False)
    parser.add_argument("-dropout", "--dropout", help="Dropout value of hidden layers", required=False)
    parser.add_argument("-weight_decay", "--weight_decay", help="Weight decay value passed into Adam", required=False)
    parser.add_argument("-optimizer", "--optimizer", help="Optimizer type", required=False)
    parser.add_argument("-init_lr", "--init_lr", help="Initial learning rate", required=False)
    parser.add_argument("-scheduler", "--scheduler", help="Learning rate scheduler", required=False)
    parser.add_argument("-step_size", "--step_size", help="step_size parameter for StepLR scheduler", required=False)
    parser.add_argument("-eta_min", "--eta_min", help="eta_min parameter for CosineAnnealingLR scheduler", required=False)
    parser.add_argument("-lr_warmup", "--lr_warmup", help="Warmup before lr scheduler", required=False)
    parser.add_argument("-warmup_epochs", "--warmup_epochs", help="Number of warmup epochs to use with lr_warmup", required=False)
    parser.add_argument("-batch_size", "--batch_size", help="Batch size", required=False)


    args, unparsed = parser.parse_known_args()
    
    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    conf['project'] = args.project
    conf['group'] = args.group
    conf['seed'] = args.seed

    if args.num_layers != None:
        conf['num_layers'] = args.num_layers
    if args.hidden_dim != None:
        conf['hidden_dim'] = args.hidden_dim
    if args.dropout != None:
        conf['dropout'] = args.dropout
    if args.weight_decay != None:
        conf['weight_decay'] = args.weight_decay
    if args.optimizer != None:
        conf['optimizer'] = args.optimizer
    if args.init_lr != None:
        conf['init_lr'] = args.init_lr
    if args.scheduler != None:
        conf['scheduler'] = args.scheduler
    if args.step_size != None:
        conf['step_size'] = args.step_size
    if args.eta_min != None:
        conf['eta_min'] = args.eta_min
    if args.lr_warmup != None:
        conf['lr_warmup'] = args.lr_warmup
    if args.warmup_epochs != None:
        conf['warmup_epochs'] = args.warmup_epochs
    if args.batch_size != None:
        conf['batch_size'] = args.batch_size
    
    return(conf)
