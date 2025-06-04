from torch.utils.dataloader import IterableDataloader


class RepairsEnvDataloader(IterableDataloader):
    def __init__(self, batch_size: int):
        super().__init__(self, batch_size=batch_size)
