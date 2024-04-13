### for debugging ###
from memory_profiler import profile

import numpy as np
import jax
from jax import numpy as jnp
from jax import random as jpr  # JAX용 random 모듈을 임포트
import time

from tensornvme import DiskOffloader

new_rng = jpr.PRNGKey(0)

def test(offloader, shape):
    
    
    
    # jax array 생성해서 offloading 한 상태로 list xs에 저장
    xs=[]
    for _ in range(10):
        x = jpr.normal(new_rng, shape=shape, dtype=np.float32)
        x = offloader.sync_write(x)
        xs.append(x)
    
    total_start_time = time.time()
    
    xs[0] = offloader.sync_read(xs[0])

    for i in range(len(xs)):        
        offloader.sync_write_events()
        xs[i] = offloader.async_write(xs[i])
        
        offloader.sync_read_events()
        if i+1 < len(xs):
            xs[i+1] = offloader.async_read(xs[i+1])
        
        
    offloader.synchronize()
    
    total_time = time.time() - total_start_time
    
    
    return total_time

def main():
    # 반복횟수 변수
    repeat = 1
    
    offloader = DiskOffloader('./offload')
    # for shape in [(1, 1024, 1024, 1024), (2, 1024, 1024, 1024), (4, 1024, 1024, 1024)]:
    for shape in [(1, 1024, 1024, 1024)]:
        acc_total_time = 0
        for i in range(repeat):
            total_time = test(offloader, shape)
            acc_total_time += total_time
        average_total_time = acc_total_time / repeat
        
        print(f"shape: {shape}, dtype: float32, size: {np.prod(shape) * 4 / 1024 / 1024} MB")   
        print(f"total_time: {average_total_time} seconds")
        
if __name__ == "__main__":
    main()
