import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize
import torch.nn.functional as F
from functorch import make_functional, vmap, vjp, jvp, jacrev
import torch
from abc import ABC, abstractmethod
from colorednoise import powerlaw_psd_gaussian
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, KERNEL_PARAMS
from sklearn.preprocessing import normalize
# from sklearn.linear_model import RidgeCV, LassoCV
from numpy.linalg import norm, pinv
import logging
from .attack import get_query
logger = logging.getLogger(__name__)


def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result


def ntk_kernel(x, y, fnet_single=None, params=None):
    x = torch.from_numpy(x).to(torch.float)
    y = torch.from_numpy(y).to(torch.float)
    res = empirical_ntk_jacobian_contraction(fnet_single, params, x, y).squeeze()
    return res.detach().numpy()


KERNEL_PARAMS['ntk'] = ('fnet_single', 'params')
PAIRWISE_KERNEL_FUNCTIONS['ntk'] = ntk_kernel


class Defense(ABC):
    """Abstract base class for defense implementation."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def respond(self, x, y):
        """Abstract method to return responses."""

    @abstractmethod
    def add_noise(self, x, y):
        """Abstract method to add perturbation."""


class DefenseClassification(Defense):
    """Base defense class for classification task.

    Parameters
    ----------
    device : str
        CPU or GPU device used for training.
    tol : float
        Clip the probability to [tol, 1-tol].
    add : bool
        Add the generated perturbation or not.
    """
    def __init__(self, *args, tol=1e-10, **kwargs):
        self.tol = tol
        super().__init__(*args, **kwargs)

    def respond(self, x, y):
        return self._scale(self._clip(self.add_noise(x, y), self.tol))

    @abstractmethod
    def add_noise(self, x, y,):
        """Calculate the perturbation and add it to the responses."""

    @staticmethod
    def _scale(y):
        """Rescale a vector such that each instance sums to one."""
        return (y.T/y.sum(axis=1)).T if len(y.shape) > 1 else y

    @staticmethod
    def _clip(y, tol):
        """Clip the probability."""
        return np.clip(y, tol, 1-tol)


class DefenseRegression(Defense):
    """Base defense class for classification task.

    Parameters
    ----------
    device : str
        CPU or GPU device used for training.
    tol : float
        Clip the probability to [tol, 1-tol].
    add : bool
        Add the generated perturbation or not.
    """
    def __init__(self, *args, lb=-np.inf, ub=np.inf, **kwargs):
        self.lb = lb
        self.ub = ub
        super().__init__(*args, **kwargs)

    def respond(self, x, y):
        return self._clip(self.add_noise(x, y), self.lb, self.ub)

    def add_noise(self, x, y):
        """Calculate the perturbation and add it to the responses."""
        return y

    @staticmethod
    def _clip(y, lb, ub):
        """Clip the probability."""
        return np.clip(y, lb, ub)


class RandomNoiseClf(DefenseClassification):
    """Add random Gaussian noise.

    Parameters
    ----------
    sigma : float
        Variance of the Gaussian distribution.
    """
    def __init__(self, *args, sigma=0.1, **kwargs):
        self.sigma = sigma
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        return y + np.random.normal(0, self.sigma, size=y.shape)


class RandomNoiseReg(DefenseRegression):
    """Add random Gaussian noise.

    Parameters
    ----------
    sigma : float
        Variance of the Gaussian distribution.
    """
    def __init__(self, sigma=0.1, *args, **kwargs):
        self.sigma = sigma
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        return y + np.random.normal(0, self.sigma, size=y.shape)


class LongRangeNoise(DefenseRegression):
    """Add random Gaussian noise.

    Parameters
    ----------
    sigma : float
        Variance of the Gaussian distribution.
    """
    def __init__(self, sigma=0.1, gamma=0.5, sort=True, *args, **kwargs):
        self.sigma = sigma
        self.gamma = gamma
        self.k = 1
        self.sort = sort
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        # TODO: generate correlated noise for sequential query.
        n = len(y)
        gamma = 1/np.power(n, self.k) if self.gamma is None else self.gamma
        noise = powerlaw_psd_gaussian(1-gamma, y.shape) if y.shape != (1,) else np.random.normal(0, 1, size=y.shape)
        if len(x.shape) == 1 and self.sort:
            noise = noise[np.argsort(x)]  # sort noise by values of x
        return y + noise*self.sigma


class Truth(Defense):
    def respond(self, x, y):
        return self.add_noise(x, y)

    def add_noise(self, x, y):
        return y


class DeceptiveNoise(DefenseClassification):
    """Add deceptive noise.

    Parameters
    ----------
    sigma : float
        Range of noise.
    beta : float
        Scale of noise.
    """
    def __init__(self, gamma=0.1, beta=1, *args, **kwargs):
        self.gamma = gamma
        self.beta = beta
        super().__init__(*args, **kwargs)

    @staticmethod
    def _r(x, gamma):
        return sigmoid(gamma * np.log(x / (1 - x))) - 1 / 2

    def add_noise(self, x, y):
        return y-self.beta*self._r(y, self.gamma)


class AdaptiveNoise(DefenseClassification):
    """Add adaptive noise.

    Parameters
    ----------
    mis_model : function
        The mis-specified model that determines noise.
    tau : float
        Threshold for suspicious inputs.
    nu : float
        Scale of noise.
    """
    def __init__(self, mis_model=None, tau=0.8, nu=-1000, device='cpu', *args, **kwargs):
        self.tau = tau
        self.nu = nu
        self.mis_model = mis_model.to(device)
        self.device = device
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        mis_target = self.mis_model(x.to(self.device)).detach().cpu().numpy()  # Outcomes of the wrong model
        excess = y.max(axis=1) - self.tau
        alpha = sigmoid(excess * self.nu)
        return (1 - alpha)[:, None] * y + alpha[:, None] * mis_target


activation = {}


def get_activation(name):
    """Get the outputs from the intermedia layers of a neural network."""
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


class Overfit(DefenseClassification):
    """Add higher order polynomial noise for NN."""
    def __init__(self, teacher, epsilon=1, device='cpu', *args, **kwargs):
        self.teacher = teacher
        self.epsilon = epsilon
        self.device = device
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        teacher = self.teacher
        teacher.train(False)
        teacher.classifier[-2].register_forward_hook(get_activation('last'))
        teacher.classifier[-1].register_forward_hook(get_activation('logits'))
        # activation = {}
        # dsg_matrix = torch.zeros(84, 10, device=device)
        # dsg_matrix[:, 0] = epsilon
        dsg_matrix = torch.rand(size=(84, 10), device=self.device) * self.epsilon
        output = teacher(x.to(self.device))
        tmp = activation['last']
        logits = activation['logits'] + torch.pow(tmp, 3) @ dsg_matrix
        return F.softmax(logits, dim=1)


class NTKBinaryNoise(DefenseClassification):
    def __init__(self, model=None, sigma=0.1, device='cpu', emp_data=None, *args, **kwargs):
        self.model = model
        self.sigma = sigma
        self.device = device
        self.newx = emp_data.to(device)
        fnet, self.params = make_functional(self.model.to(device))

        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)

        self.fnet_single = fnet_single
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        n = len(y)
        x = x.to(self.device)
        emp = len(self.newx)
        ker = self.empirical_ntk_jacobian_contraction(self.fnet_single, self.params, x, x)
        ker = ker[:, :, 1, 1].squeeze().detach().cpu().numpy()
        # ik = pinv(ker)
        alpha = self.kwargs.get('alpha', 0)
        ik = pinv(ker+alpha*np.identity(ker.shape[0]))
        pred_ker = self.empirical_ntk_jacobian_contraction(self.fnet_single, self.params, x, self.newx)
        pred_ker = pred_ker[:, :, 1, 1].squeeze().detach().cpu().numpy()
        newy = self.model(self.newx).detach().cpu().numpy()
        ey = pred_ker @ newy / emp
        eker = pred_ker @ pred_ker.T / emp
        m = ik.T @ eker @ ik
        u = y[:, 1] @ m - ik @ ey[:, 1]
        e = sol(m, 2 * u)
        e = e / norm(e)
        y[:, 1] += e * self.sigma * np.sqrt(n)
        return y

    @staticmethod
    def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = [j.flatten(2) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = [j.flatten(2) for j in jac2]

        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result


class NTKRegNoise(DefenseRegression):
    def __init__(self, model=None, sigma=0.1, device='cpu', emp_data=None, *args, **kwargs):
        self.model = model
        self.sigma = sigma
        self.device = device
        self.newx = emp_data.to(device)
        fnet, self.params = make_functional(self.model.to(device))

        def fnet_single(params, x):
            return fnet(params, x.unsqueeze(0)).squeeze(0)

        self.fnet_single = fnet_single
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        n = len(y)
        x = x.to(self.device)
        ker = self.empirical_ntk_jacobian_contraction(self.fnet_single, self.params, x, x)
        ker = ker.squeeze().detach().cpu().numpy()
        alpha = self.kwargs.get('alpha', 0)
        ik = pinv(ker+alpha*np.identity(ker.shape[0]))
        pred_ker = self.empirical_ntk_jacobian_contraction(self.fnet_single, self.params, x, self.newx)
        pred_ker = pred_ker.squeeze().detach().cpu().numpy()
        newy = self.model(self.newx).squeeze().detach().cpu().numpy()
        ey = pred_ker @ newy
        eker = pred_ker @ pred_ker.T
        m = ik.T @ eker @ ik
        u = y @ m - ik @ ey
        e = sol(m, 2 * u)
        e = e / norm(e)
        return y + e * self.sigma * np.sqrt(n)

    @staticmethod
    def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = [j.flatten(2) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = [j.flatten(2) for j in jac2]

        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result


class KRRNoise(DefenseRegression):
    def __init__(self, sigma=0.1, model=None, *args, **kwargs):
        self.sigma = sigma
        self.model = model
        self.emp = kwargs.get('emp', 1000)
        self.query_strategy = kwargs.get('query_strategy', 'IID')
        self.query_kwargs = kwargs.get('query_kwargs', {})
        self.newx = get_query(self.query_strategy, **self.query_kwargs).gen_query(self.emp)
        self.newx = self.newx.reshape(-1, 1) if len(self.newx.shape) == 1 else self.newx
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        n = len(y)
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        # emp = self.emp
        ker = pairwise_kernels(x, metric=self.kwargs['kernel'], **self.kwargs['kernel_params'])
        ik = pinv(ker + n * self.kwargs['alpha'] * np.identity(n))
        pred_ker = pairwise_kernels(x, self.newx, metric=self.kwargs['kernel'], **self.kwargs['kernel_params'])
        yhat = self.model(self.newx).squeeze()
        if isinstance(yhat, torch.Tensor):
            yhat = yhat.detach().cpu().numpy()
        ey = pred_ker @ yhat
        eker = pred_ker @ pred_ker.T
        m = ik.T @ eker @ ik
        u = y @ m - ik @ ey
        e = sol(m, 2 * u)
        # e = eigh(m)[1][:, 0]
        e = e / norm(e)
        return y + e * self.sigma * np.sqrt(n)


class LRNoise(DefenseRegression):
    def __init__(self, sigma=0.1, p=1, *args, **kwargs):
        self.sigma = sigma
        self.p = p
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        n = len(y)
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        X = np.hstack([x ** i for i in range(self.p)])
        e1 = X[:, -1]
        e2 = X @ (pinv(X.T @ X)[:, -1])
        e1, e2 = normalize([e1, e2])
        e = normalize([e1 + e2])
        return y + e.ravel() * self.sigma * np.sqrt(n)


class UniformShiftNoise(DefenseRegression):
    def __init__(self, sigma=0.1, *args, **kwargs):
        self.sigma = sigma
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        return y + np.ones_like(y) * self.sigma


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


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
    # m = m.astype(np.float32)
    # u = u.astype(np.float32)
    cons = [{'type': 'eq', 'fun': con1}]
    x0 = eigh(m)[1][:, -1]  # initial guess
    solution = minimize(obj, x0, jac=jac, constraints=cons, options={'maxiter': 1000})
    x = x0 if any(np.isnan(solution.x)) else solution.x
    logger.debug(f'Solver Status: {solution.message}')
    return x


def gram(grad_list, Y=None):
    """Calculate the gram matrix from the neural network.
    Parameters
    ----------
    grad_list : List
        List of gradients for each layer.
    Y : None or List
        None then same as grad_list.

    Returns
    -------
    numpy.ndarray
    """
    n = len(grad_list)
    m = n if Y is None else len(Y)
    gram = np.zeros((n, m))
    for i in range(n):
        u = grad_list[i]
        for j in range(m):
            if j < i and Y is None:
                gram[i, j] = gram[j, i]
            else:
                v = Y[j] if Y else grad_list[j]
                gram[i, j] = np.sum(list(map(lambda x: torch.sum(x[0]*x[1]).item(), zip(u, v))))
    return gram


def get_defense(method_name, *args, **kwargs):
    map_dict = {
        'None': Truth,
        'RandomNoiseClassification': RandomNoiseClf,
        'RandomNoiseRegression': RandomNoiseReg,
        'LongRangeNoise': LongRangeNoise,
        'KRR': KRRNoise,
        'LR': LRNoise,
        'Deceptive': DeceptiveNoise,
        'AM': AdaptiveNoise,
        'NTKBinary': NTKBinaryNoise,
        'NTKReg': NTKRegNoise,
        'UniformShift': UniformShiftNoise,
    }
    if method_name in map_dict:
        defender = map_dict[method_name](*args, **kwargs)
    else:
        raise ValueError("Invalid defender.")
    return defender
