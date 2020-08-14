import numpy as np
from numba import njit
import time
from heapq import heappush,heappop

@njit
def go_fast(a, b):
    # Function is compiled and runs in machine code

    m = a.shape[0]
    n = b.shape[0]
    f = a.shape[1]
    trace = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for k in range(f):
                trace[i, j] += a[i, k] * b[j, k]

    return trace

@njit
def go_fast_np(a, b):
    # why is it 10x slower than go_fast
    m = a.shape[0]
    n = b.shape[0]
    trace = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            trace[i, j] = a[i, :].dot(b[j, :])

    return trace


@njit
def test():
    numerator, denominator = 0, 0
    max_nn = 10
    a = [1, 2, 3,  4]

    heap = [(-a[0], 0)]
    for i, e in enumerate(a[1:], 1):
        heappush(heap, (-e, i))
        # heap.append((-e, i))

    while len(heap) > 0 and max_nn > 0:
        sim, j = heappop(heap)
        sim = -sim
        numerator += sim
        denominator += sim
        max_nn -= 1


if __name__ == '__main__':
    test()
    exit()

    x = np.arange(20000).astype(float).reshape(1000, 20)
    y = np.arange(10000).astype(float).reshape(500, 20)
    t = time.time()
    for _ in range(5):
        s = go_fast(x, y)
        prev_t = t
        t = time.time()
        print(t - prev_t)

    print('='*60)
    for _ in range(5):
        s1 = go_fast_np(x, y)
        prev_t = t
        t = time.time()
        print(t - prev_t)
