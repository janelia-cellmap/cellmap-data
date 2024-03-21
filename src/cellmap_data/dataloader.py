from torch.utils.data import DataLoader
from cellmap_data.load import transforms


class CellMapDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0): ...
