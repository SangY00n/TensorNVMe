import torch
from tensornvme import DiskOffloader

x = torch.rand(2, 2)
y = torch.rand(4, 4, 4)
offloader = DiskOffloader('./offload')
offloader.async_write(x)
# x is being offloaded in the background
offloader.sync_write_events()
# x is offloaded and the memory of x is freed
offloader.async_read(x)
# x is being restored in the background
offloader.sync_read_events()
# x is restored
offloader.async_writev([x, y])
# x and y are being offloaded in the background
offloader.synchronize()
# synchronize() will synchronize both write and read events.
offloader.async_readv([x, y])
offloader.synchronize()
# x and y are restored.
# async_writev() and async_readv() are also order sensitive