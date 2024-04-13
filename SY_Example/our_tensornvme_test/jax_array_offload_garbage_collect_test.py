from tensornvme import DiskOffloader
from jax import random
import jax.numpy as jnp
import time
import weakref
from memory_profiler import profile
    
@profile
def main():
    key = random.PRNGKey(0)
    o_xs = [random.uniform(key, shape=(1024,1024,3)) for _ in range(10)]
    
    xs_weakref = [weakref.ref(x) for x in o_xs]
    
    for x in o_xs:
        print(x.device())
    
    for x in o_xs:
        print("x:", x.shape)

    offloader = DiskOffloader('./offload')

    xs = offloader.sync_writev(o_xs)
    
    # print(o_xs)
    # del o_xs
    # # print xs_weakref
    # for x_weakref in xs_weakref:
    #     print(x_weakref())

    print("xs is offloaded")
    print()
    
    for x_weakref in xs_weakref: # 여기까지는 메모리가 할당된 상태로 남아있다가 곧 할당 해제 됨. 원래 할당 해제되는데에 시간이 좀 필요한 듯
        print(x_weakref())
        
    for x in xs:
        print("x:", x.shape)
    print()
    
    for x_weakref in xs_weakref:
        print(x_weakref())

    xs = offloader.sync_readv(xs)

    print("xs is restored")
    for x in xs:
        print("x:", x.shape)
        
    for x_weakref in xs_weakref:
        print(x_weakref())


if __name__ == '__main__':
    main()