import numpy as np
import time


def test():
    x = np.empty(0, dtype=np.float32)
    print("x's data pointer: ", x.__array_interface__['data'][0])

    start_time = time.time()
    x.resize((1024,1024,1024), refcheck=False)
    resize_time = time.time() - start_time
    print("time for resize: ", resize_time, " seconds")
    print("x's data pointer: ", x.__array_interface__['data'][0])
    print("x.shape:", x.shape)

    return resize_time

acc_time = 0
for i in range(10):
    acc_time += test()
print(f"average time for resize: {acc_time / 10} seconds")

