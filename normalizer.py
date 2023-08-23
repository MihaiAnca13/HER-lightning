import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np


class Normalizer(nn.Module):
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        super().__init__()
        self.size = size
        self.eps = nn.Parameter(torch.tensor(eps, dtype=torch.float32), requires_grad=False)
        self.default_clip_range = default_clip_range
        # get the total sum sumsq and sum count
        self.sum = nn.Parameter(torch.zeros(self.size, dtype=torch.float32), requires_grad=False)
        self.sumsq = nn.Parameter(torch.zeros(self.size, dtype=torch.float32), requires_grad=False)
        self.count = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        # get the mean and std
        self.mean = nn.Parameter(torch.zeros(self.size, dtype=torch.float32), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.size, dtype=torch.float32), requires_grad=False)
        # sync vars across processes
        self.sum.share_memory_()
        self.sumsq.share_memory_()
        self.count.share_memory_()
        self.mean.share_memory_()
        self.std.share_memory_()
        # thread locker
        self.lock = mp.Lock()

    @property
    def device(self):
        return self.count.device

    # update the parameters of the normalizer
    def update(self, v):
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).to(self.device)

        v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.sum += v.sum(dim=0)
            self.sumsq += (torch.square(v)).sum(dim=0)
            self.count[0] += v.shape[0]

    def recompute_stats(self):
        with self.lock:
            # calculate the new mean and std
            self.mean[:] = self.sum / self.count
            self.std[:] = torch.sqrt(
                torch.max(torch.square(self.eps), self.sumsq / self.count - torch.square(self.mean)))

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return torch.clamp((v - self.mean) / self.std, -clip_range, clip_range)

    def denormalize(self, v):
        return v * self.std + self.mean
