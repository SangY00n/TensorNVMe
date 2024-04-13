from tensornvme import DiskOffloader
from jax import random
import jax.numpy as jnp
import time
from memory_profiler import profile
    
@profile
def main():
    key = random.PRNGKey(0)
    xs = [random.uniform(key, shape=(1024,1024,3)) for _ in range(10)]
    for x in xs:
        print(x.device())
    xs_copy = xs.copy()
    for x in xs:
        print("x:", x.shape)

    offloader = DiskOffloader('./offload')

    xs = offloader.sync_writev(xs)

    print("xs is offloaded")
    print()
    for x in xs:
        print("x:", x.shape)
    print()

    xs = offloader.sync_readv(xs)

    print("xs is restored")
    for x in xs:
        print("x:", x.shape)
        

    for x, x_copy in zip(xs, xs_copy):
        print(jnp.allclose(x, x_copy))


if __name__ == '__main__':
    main()