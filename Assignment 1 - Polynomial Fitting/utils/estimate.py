import numpy as np
from .stochastic import RandomVariable
import matplotlib.pyplot as plt
import inspect
 
class EstimatorComparison:
    def __init__(self) -> None:
        pass

    def T(self, X):
        if self.noise_distribution == "gaussian":
            return np.sum(X)
        elif self.noise_distribution == "dependent_uniform":
            return np.max(X)
    
    def g(self, T):
        if self.noise_distribution == "gaussian":
            return T / self.N
        elif self.noise_distribution == "dependent_uniform":
            return (self.N + 1) / (2 * self.N) * T
    
    def mvue(self, n_readings, noise_distribution="gaussian"):
        self.noise_distribution = noise_distribution
        self.N = n_readings

        return lambda x: self.g(self.T(x))
    
    def compare_estimators(self, estimator1, estimator2, noise, dc_val, n_readings=1000, n_iters=1000):
        estimate_names = []

        current_frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(current_frame)[1]
        local_vars = caller_frame.frame.f_locals
    
        for name, value in local_vars.items():
            if value is estimator1:
                estimate_names.append(name)
            elif value is estimator2:
                estimate_names.append(name)

        estimates = []
        estimators = [estimator1, estimator2]
        
        for _ in range(n_iters):  
            dc = np.ones(n_readings) * dc_val  
            dc = [d + noise() for d in dc]
            
            estimates.append((estimator1(dc), estimator2(dc)))

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Noise Type: {noise.__name__}")

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plot_data = [e[i] for e in estimates]
            plt.hist(plot_data, label=estimate_names[i], bins=int(np.sqrt(n_iters)), color="orange")
            plt.title(f"Mean: {np.mean(plot_data):.2f}, Var: {np.var(plot_data):.4f}")
            plt.legend()

        plt.show()

rv = RandomVariable()

if __name__ == "__main__":
    dc_val = 5
    n_readings = 2000
    
    def dependent_uniform():
        return rv.uniform(-dc_val, dc_val)
    
    def dependent_gaussian():
        return rv.gaussian(0, dc_val)
    
    ec = EstimatorComparison()
    mean_estimator = ec.mvue(n_readings=n_readings, noise_distribution="gaussian")
    ec = EstimatorComparison()
    max_estimator = ec.mvue(n_readings=n_readings, noise_distribution="dependent_uniform")

    for noise in [dependent_uniform, rv.gaussian]:
        ec.compare_estimators(
            estimator1=mean_estimator,
            estimator2=max_estimator,
            noise=noise, 
            dc_val=dc_val,
            n_readings=n_readings, 
            n_iters=100
            )