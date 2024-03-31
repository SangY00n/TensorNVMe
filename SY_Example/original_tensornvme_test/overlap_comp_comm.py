import torch
from tensornvme import DiskOffloader

import time

offloader = DiskOffloader('./offload') # DiskOffloader 인스턴스 생성 <- offloading 할 폴더 지정

tensors = []

for _ in range(10):
    tensor = torch.rand(2, 2)
    tensors.append(tensor)
    offloader.sync_write(tensor)
    
print("Wait in 5 seconds...")
# ./offload 폴더 내에 offloading을 위한 temp 파일 1개가 생성되어 있음.
time.sleep(5)

offloader.sync_read(tensors[0])

# prefetch=1, writing tensor[i] and reading tensor[i+1]
for i, tensor in enumerate(tensors):
    offloader.sync_read_events()
    if i + 1 < len(tensors):
        offloader.async_read(tensors[i+1])
    tensor.mul_(2.0) # compute
    offloader.sync_write_events()
    offloader.async_write(tensor)
offloader.synchronize()

print("Terminate in 5 seconds...")
time.sleep(5)