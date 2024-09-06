import numpy as np
import matplotlib.pyplot as plt
from utils.stochastic import RandomVariable
from utils.polynomial import infer_polynomial, generate_noisy_polynomial, pretty_polynomial
from utils.fit import fit_line, get_fim

rv = RandomVariable()

if __name__ == "__main__":
    # data setup
    x = np.arange(-10, 10, 0.1)
    opt_params = [1, 2]
    sigma = 5

    print("Optimal Params: ", pretty_polynomial(opt_params))
    y = generate_noisy_polynomial(x, opt_params, sigma=sigma) # N(mu=0, sigma=1)

    # estimation
    estimated_params = fit_line(x, y)
    print("Estimated Params: ", pretty_polynomial(estimated_params))

    # fim and covariance
    fim = get_fim(x, estimated_params, 1)
    cov = np.linalg.inv(fim)
    crlb = np.diag(cov)
    print("CRLB:\n", crlb)

    plt.scatter(x, y)
    plt.plot(x, infer_polynomial(x, estimated_params), color="orange", label="estimated")
    plt.title(f"Actual Line: {pretty_polynomial(opt_params)}\nEstimated Line: {pretty_polynomial(estimated_params)}\nStd: {sigma}")
    plt.show()