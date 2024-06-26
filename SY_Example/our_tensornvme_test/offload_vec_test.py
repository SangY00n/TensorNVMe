from tensornvme import DiskOffloader
import numpy as np
import time
from memory_profiler import profile
    
@profile
def main():
    xs = [np.random.rand(1024,1024,3) for _ in range(10)]
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
        
    print(np.allclose(xs[0], xs_copy[0]))
    for x, x_copy in zip(xs, xs_copy):
        print(np.allclose(x, x_copy))


if __name__ == '__main__':
    main()