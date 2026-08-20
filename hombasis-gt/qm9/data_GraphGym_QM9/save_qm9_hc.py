import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp

from pathlib import Path
import sys
import os

main_dir = Path(__file__).parent.parent.parent.parent
print(main_dir)
sys.path.append(os.path.join(main_dir, 'hombasis-gt','qm9','src','utils'))
import dataset_loader as dl
from shortest_paths import ShortestPathTransform

import time
sys.path.append(os.path.join(main_dir, 'hombasis-gt','qm9','data_GraphGym_QM9'))
from CustomDataset import CustomDataset

#copied from GraphGPS/graphgps/loader/master_loader.py to avert double register
def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])

    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]

#takes in three lists of pyg elements and returns a list of three pyg InMemoryDatasets
def get_qm9_datasets(tr_graphs, val_graphs, tst_graphs):
    data_dir = f"{main_dir}/hombasis-gt/qm9/data_GraphGym_QM9/storage"
    tr_graphs = CustomDataset(tr_graphs, root=data_dir)
    val_graphs = CustomDataset(val_graphs, root=data_dir)
    tst_graphs = CustomDataset(tst_graphs, root=data_dir)
    return [tr_graphs, val_graphs, tst_graphs]

def get_qm9_graphHC_lists(root_dir, include_shortest_path=True):

    dataset_path = osp.join(root_dir, "data", "QM9")
    if include_shortest_path:
        transform_class = ShortestPathTransform(max_distance=10)
    else:
        transform_class = None

    qm9_proc_root = osp.join(dataset_path, "QM9_proc")
    tr_graphs = dl.read_qm9(
        dataset_path, "train", transform_class.transform, qm9_proc_root
    )
    val_graphs = dl.read_qm9(
        dataset_path, "valid", transform_class.transform, qm9_proc_root
    )
    tst_graphs = dl.read_qm9(
        dataset_path, "test", transform_class.transform, qm9_proc_root
    )
    num_feat = 15  # + 30 dim of all-5 vertex homcounts + 1 6 cycle
    num_pred = 13  # Class here really is used in the sense of
    return tr_graphs, val_graphs, tst_graphs, num_feat, num_pred

start_t = time.time()
tr_graphs, val_graphs, tst_graphs, num_feat, num_pred = get_qm9_graphHC_lists(f'{main_dir}/hombasis-gt/qm9')
get_qm9_graphHC_lists_t = time.time()

dataset_lists = get_qm9_datasets(tr_graphs, val_graphs, tst_graphs)
get_qm9_datasets_t = time.time()

joined_dataset = join_dataset_splits([dataset_lists[0], dataset_lists[1], dataset_lists[2]])
joined_dataset_t = time.time()


#UNCOMMENT BELOW TO SAVE GRAPH_HC
save_in_dir=f"{main_dir}/datasets/QM9-GraphHC/processed"
if not os.path.exists(save_in_dir):
    os.makedirs(save_in_dir)
torch.save(joined_dataset, os.path.join(save_in_dir,"joined.pt"))
save_t = time.time()

print(f'total time: {save_t-start_t}')