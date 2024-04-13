from tensornvme import DiskOffloader
from jax import random
import jax.numpy as jnp
import time
from memory_profiler import profile
    
@profile
def main():
    key = random.PRNGKey(0)
    x = random.uniform(key, shape=(1024,1024,3))
    print("x.device():", x.device())
    print("x:", x.shape)

    offloader = DiskOffloader('./offload')

    x = offloader.async_write(x)
    offloader.synchronize()

    print("x is offloaded")
    print()
    print("x:", x.shape)
    print()

    x = offloader.async_read(x)
    offloader.synchronize()

    print("x is restored")
    print("x:", x.shape)
    
    x = jnp.array(x)


if __name__ == '__main__':
    main()