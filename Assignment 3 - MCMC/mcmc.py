import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class GMM:
    def __init__(self, means, std_devs, weights):
        self.means = np.array(means)
        self.std_devs = np.array(std_devs)
        self.weights = np.array(weights) / np.sum(weights)
        self.num_components = len(means)

    def density(self, x):
        density = sum(w * norm.pdf(x, mu, sigma) for w, mu, sigma in zip(self.weights, self.means, self.std_devs))
        return density

class MCMC:
    def __init__(self, target_density, proposal_std=1.0):
        self.target_density = target_density 
        self.proposal_std = proposal_std
        self.samples = []

    def run(self, initial_state, num_samples, ignore_first=50):
        current_state = initial_state

        for _ in range(ignore_first):
            proposal = np.random.normal(current_state, self.proposal_std)
            acceptance_ratio = min(1, self.target_density(proposal) / self.target_density(current_state))
            if np.random.rand() < acceptance_ratio:
                current_state = proposal

        for _ in range(num_samples):
            proposal = np.random.normal(current_state, self.proposal_std)
            acceptance_ratio = min(1, self.target_density(proposal) / self.target_density(current_state))
            if np.random.rand() < acceptance_ratio:
                current_state = proposal
            self.samples.append(current_state)

        return self.samples

    def plot_histogram(self):
        plt.hist(self.samples, bins=50, density=True, alpha=0.6, label="MCMC Samples")
        plt.title("Histogram of Generated Samples")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig('plots/hist.png')
        plt.show()



if __name__ == "__main__":
    means = [0, 2, 5]
    std_devs = [1, 0.05, 0.5]
    weights = [0.5, 0.01, 0.5]
    gmm = GMM(means, std_devs, weights)

    x = np.linspace(-5, 10, 1000)
    y = np.array([gmm.density(val) for val in x])
    plt.plot(x, y, label='GMM Density', color='red')
    expr = "".join([f"{w:.2f} * N({mean}, {std}) + " for mean, std, w in zip(means, std_devs, gmm.weights)])
    plt.title(f"GMM Density: {expr[:-2]}")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig('plots/density.png')
    plt.show()

    mcmc = MCMC(target_density=gmm.density, proposal_std=1.0)
    samples = mcmc.run(initial_state=0, num_samples=10000)
    mcmc.plot_histogram()

    plt.hist(samples, bins=50, density=True, alpha=0.6, label="MCMC Samples")
    plt.plot(x, y, label='GMM Density', color='red')
    plt.title("Comparison of GMM Density and MCMC Samples")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig('plots/output.png')
    plt.show()
    