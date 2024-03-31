import torch
from tensornvme import DiskOffloader

x = torch.rand(2, 2)
y = torch.rand(4, 4, 4)
offloader = DiskOffloader('./offload')
offloader.sync_write(x)
# x is saved to a file on disk (in ./offload folder) and the memory of x is freed
offloader.sync_read(x)
# x is restored
offloader.sync_writev([x, y])
# x and y are offloaded
offloader.sync_readv([x, y])
# x and y are restored.
# sync_writev() and sync_readv() are order sensitive
# E.g. sync_writev([x, y]) and sync_writev([y, x]) are different