import os
import time

import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from numpy import NaN

from simulate_brownian import BrownianSimulation

n = 1000


def worker(sharpe_ratio, simulation_count, stop):
    brownian_simulation = BrownianSimulation(stop=stop, n=n, drift=sharpe_ratio,
                                             sigma=1, simulation_count=simulation_count)
    print("Done simulation")
    drawdown = brownian_simulation.get_drawdown()
    del brownian_simulation
    return drawdown


def calc_multiple_drawdowns(sharpe_ratio, simulation_count, stop, threads):
    results = Parallel(n_jobs=16)(delayed(worker)(sharpe_ratio, simulation_count, stop) for i in range(threads))
    drawdown1 = results[0]
    for drawdown in results[1:]:
        drawdown1.join(drawdown)
    return drawdown1


def run_simulation(sharpe_ratio, simulation_count, stop, threads, buckets, save=True):
    drawdown = calc_multiple_drawdowns(sharpe_ratio, simulation_count, stop, threads)

    arrays = [[] for i in range(buckets)]
    drawdown_depth_max = 3
    coefficient = buckets / drawdown_depth_max

    a = len(drawdown.dd_length)
    print("a is: ", a)
    for i in range(a):
        length = drawdown.dd_length[i]
        depth = drawdown.dd_depth[i]
        if 0 <= depth <= 3:
            depth_index = int(depth * coefficient) - 1
            if depth_index < 0:
                depth_index = 0
            arrays[depth_index].append(length)

    arrays = [sorted(arr) for arr in arrays]
    zeros = np.zeros(buckets)
    bucket_size = np.zeros(buckets)
    for i, arr in enumerate(arrays):
        if len(arr) == 0:
            zeros[i] = NaN
            continue
        percentile = max(int(0.95 * (len(arr))) - 1, 0)
        zeros[i] = arr[percentile]
        bucket_size[i] = len(arr)
    if save:
        path_name = "data" + str(time.time())
        os.mkdir(path_name)
        np.save(os.path.join(path_name, 'zeros.npy'), zeros)
        np.save(os.path.join(path_name, 'bucked_size.npy'), bucket_size)
    # Always override the "last"
    np.save('last-zeros.npy', zeros)
    np.save('last-bucket_size.npy', bucket_size)
    return zeros


def main():
    sharpe_ratio = 1.6
    simulation_count = 10000
    stop = 10
    threads = 1000
    buckets = 200
    x_axis = np.linspace(start=0, stop=3, num=buckets)

    # for sharpe_ratio in [0.2, 0.4, 0.8, 1.6]:
    #     zeros = run_simulation(sharpe_ratio, simulation_count, stop, threads, buckets)
    #     plt.plot(x_axis, zeros, label='SR={}'.format(sharpe_ratio))
    #     plt.legend(loc="upper left")
    #
    zeros = np.load('last-zeros.npy')
    bucket_size = np.load('last-bucket_size.npy')

    for size in bucket_size:
        print(size)

    # plt.plot(x_axis, zeros, label='SR={}'.format(sharpe_ratio))
    plt.plot(x_axis, bucket_size, label='SR={}'.format(sharpe_ratio))
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
