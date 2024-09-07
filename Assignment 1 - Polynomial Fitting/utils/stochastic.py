import random
import numpy as np
import matplotlib.pyplot as plt

class RandomVariable:
    def __init__(self) -> None:
        pass

    def uniform(self, a: int | float = 0, b: int | float = 1):
        assert b > a, "Upper Bound can't be lower!"
        return random.uniform(a, b)
    
    def exponential(self, l: int | float = 2):
        # Monte Carlo Inversion
        return - np.log(self.uniform()) / l

    def gaussian(self, mu: int | float = 0, sigma: int | float = 1):
        # Box Muller
        u = self.uniform()
        v = self.uniform()
        r = np.sqrt(-2 * np.log(u))
        theta = 2 * np.pi * v
        X = r * np.cos(theta)
        return sigma*X + mu
    
    def distribution(self, samples: list | np.ndarray, bins: int = 10):
        plt.hist(samples, bins=bins, density=True)
        plt.title("Distribution Function")
        
    def show_plot(self):
        plt.show()
    
if __name__ == "__main__":
    rv = RandomVariable()
    n = int(1e6)

    plt.subplot(1, 3, 1)
    mu, sigma = 3, 2
    samples = [rv.gaussian(mu, sigma) for _ in range(n)]
    rv.distribution(samples, bins=50)
    plt.title(f"Gaussian: mu={mu}, std={sigma}")

    plt.subplot(1, 3, 2)
    a, b = 1, 4
    samples = [rv.uniform(a, b) for _ in range(n)]
    rv.distribution(samples, bins=50)
    plt.title(f"Uniform: a={a}, b={b}")

    plt.subplot(1, 3, 3)
    l = 5
    samples = [rv.exponential(l) for _ in range(n)]
    rv.distribution(samples, bins=50)
    plt.title(f"Exponential: lambda={l}")

    rv.show_plot()