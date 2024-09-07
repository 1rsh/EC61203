
# Assignment - 1 | Estimation Theory
## Problem Statement

-   Line Fitting

-   Polynomial Fitting

-   Comparing Mean and Max Estimators in DC Estimation

## Code

### Helper Functions

**Random Variables:**

<figure>
<img src="assets/code-rv.png" />
</figure>

Distributions were generated using Monte Carlo Inversion / Box Muller
Transform from a uniform random variable.

<figure>
<img src="assets/distributions.png" />
</figure>

**Polynomials:** Generates polynomial from a list of params
(coefficients) and also generates noisy (gaussian) polynomials.

<figure>
<img src="assets/code-poly.png" />
</figure>

Plots of few polynomials with noise.

<figure>
<p><img src="assets/polynomial1.png" alt="image" /> <img src="assets/polynomial2.png"
alt="image" /> <img src="assets/polynomial3.png" alt="image" /></p>
</figure>

**Curve Fitting:**

<figure>
<img src="assets/code-curve.png" />
</figure>

Contains code for getting the Vandermonde Matrix (get_vander) and
fitting polynomial:

$$H = \begin{vmatrix}
      1       & x\_{0} & x\_{0}^{2} & \dots & x\_{0}^{n} \\ 
      1       & x\_{1} & x\_{1}^{2} & \dots & x\_{1}^{n} \\
      \vdots  & \vdots&  \vdots   &       & \vdots    \\
      1       & x\_{n} & x\_{n}^{2} & \dots & x\_{n}^{n} \\ 
    \end{vmatrix}$$
*θ* = (*H*<sup>*T*</sup>*H*)<sup>−1</sup>*H*<sup>*T*</sup>*y* = *H*<sup>†</sup>*y*

### Main Code

**Line Fitting**

<figure>
<img src="assets/code-lf.png" />
</figure>

**Polynomial Fitting**

<figure>
<img src="assets/code-pf.png" />
</figure>

**Comparing Mean and Max Estimators**

<figure>
<img src="assets/code-ce.png" />
</figure>

## Results

**Line Fitting**

<figure>
<p><img src="assets/line1.png" alt="image" /> <img src="assets/line2.png"
alt="image" /> <img src="assets/line3.png" alt="image" /></p>
</figure>

**Polynomial Fitting**

<figure>
<p><img src="assets/pf1.png" alt="image" /> <img src="assets/pf2.png" alt="image" />
<img src="assets/pf3.png" alt="image" /></p>
</figure>

**Mean vs Max Estimators** Parameter Dependent Uniform -

<figure>
<p><img src="assets/ce-du-low.png" alt="image" /> <img src="assets/ce-du-hi.png"
alt="image" /></p>
<figcaption>(a) DC=1, (b) DC=1000</figcaption>
</figure>

Gaussian -

<figure>
<p><img src="assets/ce-ga-low.png" alt="image" /> <img src="assets/ce-ga-hi.png"
alt="image" /></p>
<figcaption>(a) DC=1, (b) DC=1000</figcaption>
</figure>

## Discussion

-   I generated different types of random variables and was able to
    verify Law of Large Numbers when I varied number of samples as it
    more closely approximated the real distribution.

-   Using the random variable I had made before I modeled the input data
    (with gaussian noise)

-   Then for the first two parts, the equations for polynomial
    regression were used to get estimations for the polynomial
    coefficients

    $$H = \begin{vmatrix}
                  1       & x\_{0} & x\_{0}^{2} & \dots & x\_{0}^{n} \\ 
                  1       & x\_{1} & x\_{1}^{2} & \dots & x\_{1}^{n} \\
                  \vdots  & \vdots&  \vdots   &       & \vdots    \\
                  1       & x\_{n} & x\_{n}^{2} & \dots & x\_{n}^{n} \\ 
                \end{vmatrix}$$
    *θ* = (*H*<sup>*T*</sup>*H*)<sup>−1</sup>*H*<sup>*T*</sup>*y* = *H*<sup>†</sup>*y*

-   The polynomial corresponding to the theta value was then plotted and
    the Fisher Information Matrix and Cramer Rao Lower Bound was
    calculated.
    *C**R**L**B* = *I*<sup>−1</sup>(*θ*)

-   For comparing the mean and max estimators, first a "dependent"
    uniform random variable was created:
    *U*(−*α* \* *D**C*, *α* \* *D**C*), *α* = 0.01

-   It was interesting to see that for the dependent uniform, the mean
    estimator followed a bell-like curve and the max estimator was
    similar to a flipped exponential.

-   When the DC value was high, the max estimator was more precise in
    the case of the gaussian.