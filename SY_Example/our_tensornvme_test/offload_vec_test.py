from tensornvme import DiskOffloader
import numpy as np
import time
# from memory_profiler import profile
    
# @profile
def main():
    xs = [np.random.rand(1024,1024,3) for _ in range(10)]
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


if __name__ == '__main__':
    main()