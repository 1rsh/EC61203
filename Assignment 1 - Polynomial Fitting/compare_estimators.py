from utils.estimate import EstimatorComparison
from utils.stochastic import RandomVariable

rv = RandomVariable()

dc_val = 5
n_readings = 20

def dependent_uniform():
    return rv.uniform(-dc_val, dc_val)

def dependent_gaussian():
    return rv.gaussian(0, 3*dc_val)

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
        n_iters=1000
        )