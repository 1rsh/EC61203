import numpy as np
from .stochastic import RandomVariable
import matplotlib.pyplot as plt

rv = RandomVariable()

def infer_polynomial(x: int | float | np.ndarray, params: list):
    cexp = 1
    val = 0
    for p in params[::-1]:
        val += p * cexp
        cexp *= x
    return val

def generate_noisy_polynomial(x: int | float | np.ndarray, params: list, mu=0, sigma=20):
    y = infer_polynomial(x, params)
    y = [_y + rv.gaussian(mu=mu, sigma=sigma) for _y in y]
    return np.array(y)

def pretty_polynomial(params):
    params = [round(p, 2) for p in params]
    
    operand = " + "
    to_show = []
    for i, p in enumerate(params[::-1]):
        if i==1:
            to_show.append(operand+"x")
        elif i:
            to_show.append(operand+str(i)+"^x")
        if p!=1:
            to_show.append(str(abs(p))[::-1])
        operand = " + " if p>0 else " - "
    
    if p<0:
        to_show.append(" - ")
    return ("".join(to_show))[::-1]


if __name__ == "__main__":
    x = np.arange(-10, 10, 0.1)
    params = [4, -100]
    mu = 0
    sigma = 5
    print(pretty_polynomial(params))
    y = generate_noisy_polynomial(x, params, mu=mu, sigma=sigma)
    plt.scatter(x, y)

    plt.title(f"{pretty_polynomial(params)} with noise ~ N({mu},{sigma})")
    plt.show()