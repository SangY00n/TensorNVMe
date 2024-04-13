### for debugging ###
from memory_profiler import profile

import numpy as np
import jax
from jax import numpy as jnp
from jax import random as jpr  # JAX용 random 모듈을 임포트
import time

from tensornvme import DiskOffloader

# jax.config.update('jax_platform_name', 'cpu')
devices=jax.devices() 
# 사용 가능한 디바이스 출력
for device in devices:
    print(device) # TFRT_CPU_0

@profile
def testLoad(n):
    print(f"test for {n} x {n} x 3 array")
    
    # jax array 생성
    new_rng = jpr.PRNGKey(0)
    x = jpr.normal(new_rng, (n*n*3,), dtype=np.float32).reshape(n,n,3)
    # np array 생성
    # x = np.random.rand(n,n,3).astype(np.float32)
    print("array shape: ", x.shape)
    
    offloader = DiskOffloader('./offload')


    start_time = time.time()
    x = offloader.sync_write(x)
    print("time for write: ", time.time() - start_time, " seconds")
    print("array shape: ", x.shape)
    
    start_time = time.time()
    x = offloader.sync_read(x)
    print("time for read: ", time.time() - start_time, " seconds")
    
    print("array shape: ", x.shape)
    
if __name__ == "__main__":
    for n in [2048, 4096, 8192, 16384]:
        testLoad(n)
