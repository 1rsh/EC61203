import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class LDA:
  def __init__(self):
    pass

  def _calc_scatter(self, X, y):
    d = X.shape[1]
    S_w = np.zeros((d, d))
    X_0, X_1 = X[y==0], X[y==1]
    mu_0, mu_1 = np.mean(X_0, axis=0), np.mean(X_1, axis=0)
    S_w = (X_0 - mu_0).T @ (X_0 - mu_0) + (X_1 - mu_1).T @ (X_1 - mu_1)
    return S_w, mu_0, mu_1

  def fit_transform(self, X, y):
    self.fit(X, y)
    return self.transform(X)

  def transform(self, X):
    return X @ self.w

  def fit(self, X, y):
    S_w, mu_0, mu_1 = self._calc_scatter(X, y)
    S_w_inv = np.linalg.inv(S_w)
    self.w = S_w_inv @ (mu_0 - mu_1)
    return

  def plot2d(self, X, y):   
    X_transformed = self.fit_transform(X, y)
    plt.scatter(X_transformed, np.zeros_like(X_transformed), c=y)
    plt.show()

if __name__ == "__main__":
    lda = LDA()
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_classes=2, random_state=42)
    lda.plot2d(X, y)