import torch
import time
from memory_profiler import profile

N=2048

@profile
def test():
    x = torch.randn((N,N,N), dtype=torch.float32)
    x.storage().resize_(0)
    
    # x = torch.empty(0, dtype=torch.float32)
    
    print("x.shape:", x.shape)
    print("x's data pointer: ", x.storage().data_ptr())

    start_time = time.time()
    x.storage().resize_(N*N*N)
    # x.resize((1024,1024,1024))
    resize_time = time.time() - start_time
    print("time for resize: ", resize_time, " seconds")
    print("x's data pointer: ", x.storage().data_ptr())
    print("x.shape:", x.shape)

    return resize_time

repeat = 3
acc_time = 0
for i in range(repeat):
    acc_time += test()
    print("--------------------------------------------------")
print(f"average time for resize: {acc_time / repeat} seconds")