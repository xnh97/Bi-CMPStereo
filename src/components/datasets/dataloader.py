import torch.utils.data

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):# kwargs : dataset, batch_size, collect_fn, num_workers, pin_memory, shuffle, drop_last
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
    
    def __len__(self):
        length = len(self.batch_sampler.sampler)
        return length
    
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler
    
    def __iter__(self):
        while True:
            yield from iter(self.sampler)