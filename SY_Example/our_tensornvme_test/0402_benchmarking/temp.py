### for debugging ###
from memory_profiler import profile

import numpy as np
import jax
from jax import numpy as jnp
from jax import random as jpr  # JAX용 random 모듈을 임포트
import time

from tensornvme import DiskOffloader

import copy

new_rng = jpr.PRNGKey(0)

@profile
def test(offloader, shape):
    xs=[jpr.normal(new_rng, shape=shape, dtype=np.float32) for _ in range(10)]
    ys = copy.deepcopy(xs)
    ys = np.asarray(ys)
    
    start_time = time.time()
    for i in range(5):
        xs[i] = offloader.async_write(xs[i])
        offloader.synchronize()
        print(f"xs[{i}] is offloaded... sleep 5 seconds...")
        # time.sleep(5)
    print("time for offloading: ", time.time() - start_time, " seconds")
    
    start_time = time.time()
    for i in range(5):
        xs[i] = offloader.async_read(xs[i])
        offloader.synchronize()
        print(f"xs[{i}] is restored... sleep 5 seconds...")
        # time.sleep(5)
    print("time for restoring: ", time.time() - start_time, " seconds")
    
    start_time = time.time()
    for i in range(5):
        xs[i+5] = offloader.async_write(xs[i+5])
        offloader.synchronize()
        print(f"xs[{i+5}] is offloaded... sleep 5 seconds...")
        # time.sleep(5)
    print("time for offloading: ", time.time() - start_time, " seconds")
    
    start_time = time.time()
    for i in range(5):
        xs[i+5] = offloader.async_read(xs[i+5])
        offloader.synchronize()
        print(f"xs[{i+5}] is restored... sleep 5 seconds...")
        # time.sleep(5)
    print("time for restoring: ", time.time() - start_time, " seconds")
    
    print(np.allclose(xs, ys))
    
    
    

def main():    
    offloader = DiskOffloader('./offload')
    for shape in [(256, 1024, 1024)]:
        test(offloader, shape)
        
if __name__ == "__main__":
    main()
