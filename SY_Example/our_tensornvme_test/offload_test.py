from tensornvme import DiskOffloader
import numpy as np
import time
from memory_profiler import profile
    
@profile
def main():
    x = np.random.rand(1024,1024,3)
    print("x:", x.shape)

    offloader = DiskOffloader('./offload')

    x = offloader.sync_write(x)

    print("x is offloaded")
    print()
    print("x:", x.shape)
    print()

    x = offloader.sync_read(x)

    print("x is restored")
    print("x:", x.shape)


if __name__ == '__main__':
    main()