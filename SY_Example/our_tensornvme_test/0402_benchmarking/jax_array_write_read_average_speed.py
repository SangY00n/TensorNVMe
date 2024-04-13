### for debugging ###
from memory_profiler import profile

import numpy as np
import jax
from jax import numpy as jnp
from jax import random as jpr  # JAX용 random 모듈을 임포트
import time

from tensornvme import DiskOffloader

new_rng = jpr.PRNGKey(0)

def test(offloader, shape,i):
    
    # jax array 생성
    
    x = jpr.normal(new_rng, shape=shape, dtype=np.float32)
    print("iter: ", i, "array shape: ", x.shape)
    
    start_time = time.time()
    x = offloader.async_write(x)
    offloader.synchronize()
    time_for_write = time.time() - start_time
    
    start_time = time.time()
    x = offloader.async_read(x)
    offloader.synchronize()
    time_for_read = time.time() - start_time
    # print("array shape: ", x.shape)
    
    return time_for_write, time_for_read

def main():
    # 반복횟수 변수
    repeat = 10
    
    offloader = DiskOffloader('./offload')
    for shape in [(1, 1024, 1024, 1024), (2, 1024, 1024, 1024), (4, 1024, 1024, 1024)]:
        acc_time_for_write = 0
        acc_time_for_read = 0
        for i in range(repeat):
            time_for_write, time_for_read = test(offloader, shape, i)
            acc_time_for_write += time_for_write
            acc_time_for_read += time_for_read
        average_time_for_write = acc_time_for_write / repeat
        average_time_for_read = acc_time_for_read / repeat
        
        print(f"shape: {shape}, dtype: float32, size: {np.prod(shape) * 4 / 1024 / 1024} MB")   
        print(f"average time for write: {average_time_for_write} seconds")
        print(f"average time for read: {average_time_for_read} seconds")
        
if __name__ == "__main__":
    main()
