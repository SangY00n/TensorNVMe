### for debugging ###
from memory_profiler import profile

import numpy as np
import jax
from jax import numpy as jnp
from jax import random as jpr  # JAX용 random 모듈을 임포트
import time
import copy

from tensornvme import DiskOffloader

new_rng = jpr.PRNGKey(0)
array_num = 3

@profile
def test(offloader, shape):
    xs=[]
    ys=[]
    for _ in range(array_num):
        x = jpr.normal(new_rng, shape=shape, dtype=np.float32)
        ys.append(np.asarray(copy.deepcopy(x)))
        x = offloader.sync_write(x)
        xs.append(x)
    
    
    for i in range(array_num):
        xs[i] = offloader.sync_read(xs[i])
    
    print("xs[0]'s shape", xs[0].shape)
    print("ys[0]'s shape", ys[0].shape)
    print("All Close: ", np.allclose(xs, ys))

def main():
    offloader = DiskOffloader('./offload')
    
    shape = (256, 1024, 1024) # 256*1024*1024 * 4bytes = 1GB size
    
    test(offloader, shape)
    
    print(f"shape: {shape}, dtype: float32, size: {np.prod(shape) * 4 / 1024 / 1024} MB")   
        
if __name__ == "__main__":
    main()
