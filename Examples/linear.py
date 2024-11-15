import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, t, x, H=lambda t: np.vstack([np.ones_like(t), t]).T):
        """
        t -> independent
        x -> dependent
        """
        H = H(t)

        theta = np.linalg.inv(H.T @ H) @ H.T @ x
        self.theta = theta
    
    def transform(self, x: int | float | np.ndarray):
        cexp = 1
        val = 0
        for p in self.theta:
            val += p * cexp
            cexp *= x
        return val
    
    def plot(self, t, x, H=lambda t: np.vstack([np.ones_like(t), t]).T):
        self.fit(t, x, H)

        plt.scatter(t, x, color="lightblue")
        plt.plot(t, self.transform(t))
        plt.show()

class RidgeRegression(LinearRegression):
    def fit(self, t, x, H=lambda t: np.vstack([np.ones_like(t), t]).T, lamda=0.1):
        H = H(t)

        theta = np.linalg.inv(H.T @ H + lamda * np.eye(H.shape[1])) @ H.T @ x
        self.theta = theta
    
    def plot(self, t, x, H=lambda t: np.vstack([np.ones_like(t), t]).T, lamda=0.1):
        self.fit(t, x, H, lamda)

        plt.scatter(t, x, color="lightblue")
        plt.plot(t, self.transform(t))
        plt.show()
    
if __name__ == "__main__":
    lr = RidgeRegression()

    sample_t = np.arange(-10, 10, 1)
    sample_x = 2 * sample_t**2 + 3 * sample_t + np.random.normal(0, 10)

    N = len(sample_t)

    lr.plot(sample_t, sample_x, H=lambda t: np.vstack([t**i for i in range(N)]).T, lamda=1e7)
