from tensornvme import DiskOffloader
from jax import random
import jax.numpy as jnp
import time
from memory_profiler import profile
    
@profile
def main():
    key = random.PRNGKey(0)
    xs = [random.uniform(key, shape=(256,1024,1024)) for _ in range(10)]
    for x in xs:
        print(x.device())
        print(x.dtype)
    xs_copy = xs.copy()
    for x in xs:
        print("x:", x.shape)

    offloader = DiskOffloader('./offload')

    xs = offloader.sync_writev(xs)
    # xs = offloader.async_writev(xs)
    # offloader.synchronize()

    print("xs is offloaded")
    print()
    time.sleep(15)
    
    for x in xs:
        print("x:", x.shape)
    print()

    xs = offloader.sync_readv(xs)
    # xs = offloader.async_readv(xs)
    # offloader.synchronize()

    print("xs is restored")
    for x in xs:
        print("x:", x.shape)
        

    for x, x_copy in zip(xs, xs_copy):
        print(jnp.allclose(x, x_copy))
        # print(x[0][0][:10], x_copy[0][0][:10])
        # print(x[-1][-1][-1], x_copy[-1][-1][-1])


if __name__ == '__main__':
    main()