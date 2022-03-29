import numpy as np
import matplotlib.pyplot as plt

T = 1
n = 10000
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
    x = np.linspace(start=0, stop=1, num=brownian_simulation.n + 1)
    plt.plot(x, brownian[:, 0])
    plt.show()


def main():
    simple_example()
    # sharpe_ratios = np.linspace(start=0, stop=10, num=1000)
    # for sharpe_ratio in sharpe_ratios:


if __name__ == "__main__":
    main()
