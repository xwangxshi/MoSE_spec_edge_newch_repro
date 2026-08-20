import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data


class CustomDataset(InMemoryDataset):
    def __init__(self,listOfDataObjects, root=None):
        super().__init__(root=root)
        self.data, self.slices = self.collate(listOfDataObjects)
    