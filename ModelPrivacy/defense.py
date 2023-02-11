import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize
import torch.nn.functional as F
import torch
from abc import ABC, abstractmethod
from colorednoise import powerlaw_psd_gaussian
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize
from numpy.linalg import norm, pinv
import logging
from .attack import get_query
logger = logging.getLogger(__name__)


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

# class DefenseClassfication(Defense):
#     """Base defense class for classification task.
#
#     Parameters
#     ----------
#     dataloader : torch.utils.data.Dataset
#         Dataset loader.
#     teacher : function
#         Teacher model.
#     device : str
#         CPU or GPU device used for training.
#     tol : float
#         Clip the probability to [tol, 1-tol].
#     add : bool
#         Add the generated perturbation or not.
#     """
#     def __init__(self, dataloader, teacher, device='cpu', tol=1e-10, add=True):
#         self.data = []  # retrieve the predictors of the training sample
#         self.target = []  # retrieve the outcome of the teacher model
#         self.true_target = []  # retrieve the outcome of the training sample
#         teacher.train(False)
#         for data, target in dataloader:
#             output = teacher(data.to(device))
#             self.data.append(data)
#             self.target.append(output)
#             self.true_target.append(target.to(device))
#         self.data = torch.vstack(self.data)
#         self.target = torch.vstack(self.target).squeeze().detach()
#         self.true_target = torch.hstack(self.true_target).squeeze().detach()
#         self.device = device
#         self.tol = tol
#
#         if add:
#             self.y = self._scale(self._clip(self.add_noise(), self.tol))
#         else:
#             self.y = self.target
#
#     def add_noise(self):
#         """Calculate the perturbation and add it to the responses."""
#         return self.target
#
#     def sample(self, n=None, batch_size=128, type='defended'):
#         """Sample the dataset.
#
#         Parameters
#         ----------
#         n : int
#             Sample size.
#         batch_size : int
#             Batch size.
#         type : str
#             defended : sample the perturbed dataset
#             teacher : sample the undefended dataset
#             All others : sample the original dataset
#
#         Returns
#         -------
#         torch.utils.data.DataLoader
#         """
#         if n is None:
#             n = self.data.shape[0]
#         if type == 'defended':
#             attack_set = SimData(self.data[:n], self.y[:n])
#         elif type == 'teacher':
#             attack_set = SimData(self.data[:n], self.target[:n])
#         else:
#             attack_set = SimData(self.data[:n], self.true_target[:n])
#         return torch.utils.data.DataLoader(attack_set, batch_size=batch_size, shuffle=True)
#
#     @staticmethod
#     def _scale(y):
#         """Rescale a vector such that each instance sums to one."""
#         return (y.T/y.sum(dim=1)).T if len(y.shape) > 1 else y
#
#     @staticmethod
#     def _clip(y, tol):
#         """Clip the probability."""
#         return torch.clip(y, tol, 1-tol)


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
        self.sort = sort
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        # TODO: generate correlated noise for sequential query.
        noise = powerlaw_psd_gaussian(1-self.gamma, y.shape) if y.shape != (1,) else np.random.normal(0, 1, size=y.shape)
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


class KRRNoise(DefenseRegression):
    def __init__(self, sigma=0.1, model=None, *args, **kwargs):
        self.sigma = sigma
        self.model = model
        self.emp = kwargs.get('emp', 3000)
        self.query_strategy = kwargs.get('query_strategy', 'IID')
        self.query_kwargs = kwargs.get('query_kwargs', {})
        self.newx = get_query(self.query_strategy, **self.query_kwargs).gen_query(self.emp)
        super().__init__(*args, **kwargs)

    def add_noise(self, x, y):
        n = len(y)
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        emp = self.emp
        ker = pairwise_kernels(x, metric=self.kwargs['kernel'])
        ik = pinv(ker + self.kwargs['alpha'] * np.identity(n))
        pred_ker = pairwise_kernels(x, self.newx.reshape(-1, 1), metric=self.kwargs['kernel'])
        ey = pred_ker @ self.model(self.newx) / emp
        eker = pred_ker @ pred_ker.T / emp
        m = ik.T @ eker @ ik
        u = y @ m - ik @ ey
        e = self.sol(m, 2 * u)
        e = e / norm(e)
        return y + e * self.sigma * np.sqrt(n)

    @staticmethod
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
        x0 = eigh(m)[1][:, -1]  # initial guess
        solution = minimize(obj, x0, jac=jac, constraints=cons)
        x = solution.x
        return x


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


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


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
    if method_name == 'None':
        defender = Truth(*args, **kwargs)
    elif method_name == 'RandomNoiseClassification':
        defender = RandomNoiseClf(*args, **kwargs)
    elif method_name == 'RandomNoiseRegression':
        defender = RandomNoiseReg(*args, **kwargs)
    elif method_name == 'LongRangeNoise':
        defender = LongRangeNoise(*args, **kwargs)
    elif method_name == 'KRR':
        defender = KRRNoise(*args, **kwargs)
    elif method_name == 'LR':
        defender = LRNoise(*args, **kwargs)
    elif method_name == 'Deceptive':
        defender = DeceptiveNoise(*args, **kwargs)
    elif method_name == 'AM':
        defender = AdaptiveNoise(*args, **kwargs)
    else:
        raise ValueError("Invalid defender.")
    return defender
