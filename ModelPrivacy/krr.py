"""Example of kernel ridge regression."""
import numpy as np
from numpy.linalg import norm, pinv, eigh
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import mean_squared_error


def f(x):
    """Target function."""
    return x-1.2*x**2-0.8*x**3+0.6*np.cos(2*np.pi*x)


def sol(m, u):
    """Solve the best perturbation direction."""
    def obj(x):
        """Objective function"""
        return -x @ m @ x - u @ x

    def jac(x):
        """Jacobian of the objective"""
        return -2 * m @ x - u

    def con1(x):
        """Constraint"""
        return x @ x - 1

    cons = [{'type': 'eq', 'fun': con1}]
    x0 = eigh(m)[1][:, 0]  # initial guess
    solution = minimize(obj, x0, jac=jac, constraints=cons)
    x = solution.x
    return x


# Compare three defense machanisms
nrep = 1
res_kr = np.zeros((nrep, 3))
ans = np.zeros(nrep)
emp = 3000  # Monte Carlo sample size
rng = np.random.RandomState(0)
for i in range(nrep):
    data = np.random.normal(0, 1, 1000)  # All data
    newx = np.random.normal(0, 1, emp)  # Sample for Monte Carlo
    data.sort()
    data = data.reshape(-1, 1)
    target = f(data).ravel()  # simulated nonlinear func

    # Split training and test
    training_sample_indices = rng.choice(np.arange(0, 1000), size=40, replace=False)
    training_data = data[training_sample_indices]
    training_noisy_target = target[training_sample_indices] + 0.3 * rng.randn(
        len(training_sample_indices)
    )
    an = mean_squared_error(target[training_sample_indices], training_noisy_target)  # utility loss
    ans[i] = an

    kernel_ridge = KernelRidge(alpha=1e-3, kernel='rbf', gamma=1)
    # No defense
    kernel_ridge.fit(training_data, target[training_sample_indices])
    predictions_nn = kernel_ridge.predict(data)
    res_kr[i, 0] = mean_squared_error(target, predictions_nn)

    # Random noise
    kernel_ridge.fit(training_data, training_noisy_target)
    predictions_nn = kernel_ridge.predict(data)
    res_kr[i, 1] = mean_squared_error(target, predictions_nn)

    # Best perturbation
    ker = pairwise_kernels(training_data, metric=kernel_ridge.kernel)
    ik = pinv(ker + kernel_ridge.alpha * np.identity(40))
    pred_ker = pairwise_kernels(training_data, newx.reshape(-1, 1), metric=kernel_ridge.kernel)
    ey = pred_ker@f(newx)/emp
    eker = pred_ker@pred_ker.T/emp
    m = ik.T@eker@ik
    u = target[training_sample_indices]@m - ik@ey
    e = sol(m, 2 * u)
    e = np.sqrt(40 * an) * e / norm(e)

    kernel_ridge.fit(training_data, target[training_sample_indices] + e)
    predictions_nn = kernel_ridge.predict(data)
    res_kr[i, 2] = mean_squared_error(target, predictions_nn)

print(res_kr)
