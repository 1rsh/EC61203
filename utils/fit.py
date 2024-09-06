import numpy as np

def get_vander(x, degree):
    vander = []
    for i in range(degree+1):
        vander.append(x**i)
    return np.vstack(vander).T

def fit_polynomial(x, y, degree):
    H = get_vander(x, degree)
    params = np.linalg.inv(H.T @ H) @ H.T @ y
    return params[::-1]

def fit_line(x, y):
    H = np.vstack([np.ones_like(x), x]).T
    params = np.linalg.inv(H.T @ H) @ H.T @ y
    return params[::-1]

def get_fim(x, est_params, est_sigma):
    degree = len(est_params) - 1
    I = np.zeros((degree + 1, degree + 1))

    for i in range(degree + 1):
        for j in range(degree + 1):
            grad_i = x ** i
            grad_j = x ** j
            I[i, j] = np.sum(grad_i * grad_j) / est_sigma**2
    
    return I

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    print(get_vander(x, 2))
    y = 2 * x + 1
    est_params = fit_line(x, y)
    print(est_params)
    print(get_fim(x, est_params, 1))