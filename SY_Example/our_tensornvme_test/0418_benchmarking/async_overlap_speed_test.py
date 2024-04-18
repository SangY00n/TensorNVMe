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
array_num = 10

@profile
def test(offloader, shape):    
    xs=[]
    for _ in range(array_num):
        x = jpr.normal(new_rng, shape=shape, dtype=np.float32)
        x = offloader.sync_write(x)
        xs.append(x)
    
    read_time_list = []
    write_time_list = []
    total_start_time = time.time()
    
    xs[0] = offloader.sync_read(xs[0])
    read_cur_time = time.time()
    read_time_list.append(read_cur_time-total_start_time)
    read_pre_time = time.time()
    for i in range(len(xs)):
        offloader.sync_write_events()
        if i>0:
            write_cur_time = time.time()
            write_time_list.append(write_cur_time-write_pre_time)
            write_pre_time = time.time()
        else:
            write_pre_time = time.time()
        xs[i] = offloader.async_write(xs[i])
        
        
        offloader.sync_read_events()
        if i>0:
            read_cur_time = time.time()
            read_time_list.append(read_cur_time-read_pre_time)
            read_pre_time = time.time()
        if i+1 < len(xs):
            xs[i+1] = offloader.async_read(xs[i+1])
        
        
    offloader.synchronize()
    
    total_time = time.time() - total_start_time
    print("Total time: ", total_time)
    
    print("read_time_list: ", read_time_list)
    print("write_time_list: ", write_time_list)
    
    return total_time

def main():
    offloader = DiskOffloader('./offload')
    
    repeat = 1
    shape = (256, 1024, 1024) # 256*1024*1024 * 4bytes = 1GB size
    
    acc_total_time = 0
    for i in range(repeat):
        total_time = test(offloader, shape)
        acc_total_time += total_time
    average_total_time = acc_total_time / repeat
    
    print(f"shape: {shape}, dtype: float32, size: {np.prod(shape) * 4 / 1024 / 1024} MB")   
    print(f"total_time: {average_total_time} seconds")
        
if __name__ == "__main__":
    main()
