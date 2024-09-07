import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class MDA:
  def __init__(self):
    pass

  def _calc_scatter(self, X, y):
    d = X.shape[1]
    classes = np.unique(y)

    S_w = np.zeros((d, d))

    for c in classes:
      X_c = X[y == c] # Nc * d
      mu_c = np.mean(X_c, axis=0) # 1 * d
      S_w += (len(X_c))* (X_c - mu_c).T @ (X_c - mu_c) # d * d

    mu = np.mean(X, axis=0)
    S_b = np.zeros((d, d))
    for c in classes:
        X_c = X[y==c] # Nc * d
        mu_c = np.mean(X_c, axis=0) # 1 * d
        mu_diff = (mu_c - mu).reshape(-1, 1) # d * 1
        S_b += len(X_c) * (mu_diff @ mu_diff.T) #  d * d

    return S_w, S_b

  def fit_transform(self, X, y, op_dim):
    self.fit(X, y, op_dim)
    return self.transform(X)

  def transform(self, X):
    return X @ self.eigenvectors

  def fit(self, X, y, op_dim):
    S_w, S_b = self._calc_scatter(X, y)
    S_w_inv = np.linalg.inv(S_w)
    w = S_w_inv @ S_b # d * d
    eigenvalues, eigenvectors = np.linalg.eig(w) # d eigenvectors (input dim)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx][:op_dim]
    self.eigenvectors = eigenvectors[:, idx][:, :op_dim]

    return

  def plot2d(self, X, y):
    X_transformed = self.fit_transform(X, y, op_dim=2)
    x1 = X_transformed[:, 0]
    x2 = X_transformed[:, 1]
    plt.scatter(x1, x2, c=y)
    plt.show()

if __name__ == "__main__":
    mda = MDA()
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_classes=4)
    mda.plot2d(X, y)