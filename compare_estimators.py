import numpy as np
import matplotlib.pyplot as plt
from utils.stochastic import RandomVariable

def noisy_dc(dc, noise):
    x = np.arange(-10, 10, 0.1)
    y = [dc] * len(x)
    y = [_y + noise() for _y in y]
    return np.array(y)

rv = RandomVariable()
n_iters = int(1e4)
dc = 1000
alpha = 1e-2

def dependent_uniform():
    return rv.uniform(-alpha*dc, alpha*dc)

for noise in [dependent_uniform, rv.gaussian]:
    estimates = {"mean": [], "max": []}

    for _ in range(n_iters):
        y = noisy_dc(dc, noise)
        estimates["mean"].append(np.mean(y))
        estimates["max"].append(np.max(y))
    
    biases = {"mean": 0, "max": 0}
    stds = {"mean": 0, "max": 0}

    for k in biases.keys():
        biases[k] = np.mean(estimates[k]) - dc
        stds[k] = np.std(estimates[k])
    
    plt.subplot(1, 2, 1)
    plt.hist(estimates["mean"], label="mean", bins=50)
    plt.title(f"mean | bias:{biases['mean']:.4f}, std:{stds['mean']:.4f}")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(estimates["max"], label="max", bins=50)
    plt.title(f"max | bias:{biases['max']:.4f}, std:{stds['max']:.4f}")
    plt.legend()
    plt.suptitle(str(noise.__name__))
    plt.show()
