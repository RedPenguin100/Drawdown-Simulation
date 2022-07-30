import time

from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

T = 10
n = 1000
sigma = 1


class Drawdown:
    def __init__(self, pnl_max, dd_length, dd_depth):
        self.pnl_max = pnl_max
        self.dd_length = dd_length
        self.dd_depth = dd_depth

    def __str__(self):
        return f"The Pnl_max is: {self.pnl_max}\n" \
               f"The drawdown length is: {self.dd_length}\n" \
               f"The drawdown depth is: {self.dd_depth}."

    def join(self, other: 'Drawdown'):
        self.pnl_max = np.concatenate((self.pnl_max, other.pnl_max))
        self.dd_length = np.concatenate((self.dd_length, other.dd_length))
        self.dd_depth = np.concatenate((self.dd_depth, other.dd_depth))


class BrownianSimulation:
    def __init__(self, stop, n, sigma, drift=0, simulation_count=2):
        # Assume start = 0

        # For the sake of calculation drift=sharpe ratio because we assume we are
        # already risk-neutral and the sigma = 1.
        if sigma == 1:
            self.sharpe_ratio = drift
        else:
            raise ValueError("Currently only sigma=1 is supported.")
        self.stop = stop
        self.n = n
        self.sigma = sigma
        self.drift = drift
        self.simulation_count = simulation_count
        dt = stop / n
        drift_array = np.cumsum(np.ones((n, simulation_count)) * dt * drift, axis=0)
        normal = np.random.normal(0, scale=np.sqrt(dt) * (sigma ** 2), size=(n, simulation_count))
        self.brownians = np.insert(drift_array + np.cumsum(normal, axis=0), 0, 0, axis=0)

    def get_drawdown(self):
        argmax = np.argmax(self.brownians, axis=0)
        pnl_max = self.brownians[argmax, np.arange(self.simulation_count)]
        time_of_max = (argmax / self.n) * self.stop
        dd_length = self.stop - time_of_max
        dd_depth = pnl_max - self.brownians[-1]
        return Drawdown(pnl_max, dd_length, dd_depth)


def simple_example():
    brownian_simulation = BrownianSimulation(stop=T, n=n, drift=1, sigma=sigma)
    brownian = brownian_simulation.brownians
    drawdown = brownian_simulation.get_drawdown()
    print(drawdown)
    x = np.linspace(start=0, stop=T, num=brownian_simulation.n + 1)
    plt.plot(x, brownian[:, 0])
    plt.show()


def worker(sharpe_ratio):
    print("working on sharpe ratio: ", sharpe_ratio)
    simulations = 1000
    nth_percentile = int(0.95 * simulations)

    # print(f"Simulating sharpe ratio: {sharpe_ratio}")
    brownian_simulation = BrownianSimulation(stop=T, n=n, drift=sharpe_ratio, sigma=1,
                                             simulation_count=simulations)
    drawdown = brownian_simulation.get_drawdown()
    drawdown.dd_length.sort()
    drawdown.dd_depth.sort()
    nth_percentile_depth = drawdown.dd_depth[nth_percentile]  # 5th percentile
    nth_percentile_length = drawdown.dd_length[nth_percentile]  # 5th percentile
    return nth_percentile_length, nth_percentile_depth


def multiple_drawdowns():
    N = 10001
    sharpe_ratios = np.linspace(start=0, stop=10, num=N)
    lengths = []
    depths = []
    start = time.time()
    results = Parallel(n_jobs=16)(delayed(worker)(sharpe_ratio) for sharpe_ratio in sharpe_ratios)
    print(results)
    for res in results:
        lengths.append(res[0])
        depths.append(res[1])
    stop = time.time()
    print(f"It took: {stop - start} seconds")
    fig, ax = plt.subplots()
    ax.text(6, 4, 'N= {}'.format(N))
    plt.plot(sharpe_ratios, lengths, label='lengths')
    plt.plot(sharpe_ratios, depths, label='depths')
    plt.legend(loc="upper left")
    plt.show()


def main():
    # simple_example()

    multiple_drawdowns()


if __name__ == "__main__":
    main()
