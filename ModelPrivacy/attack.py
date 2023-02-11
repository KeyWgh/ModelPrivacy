from abc import ABC, abstractmethod
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from scipy.special import legendre
from torch.utils.data import Dataset
import logging
from .nn import SimData, run
logger = logging.getLogger(__name__)


class AbstractQuery(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def gen_query(self, *args):
        """Abstract method to generate queries."""

    @abstractmethod
    def get_data(self, n, server):
        """Abstract method to retrieve data."""


class SequentialQuery(AbstractQuery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self, n, server):
        return self.gen_query(n, server)

    def gen_query(self, n, server):
        x = []
        y = []
        for i in range(n):
            x_new = self.gen_next()
            y_new = server.respond(x_new)
            x.append(x_new)
            y.append(y_new)

        return np.array(x), np.array(y)

    @abstractmethod
    def gen_next(self):
        """Abstract method to generate next query."""


class RandomSequentialQuery(SequentialQuery):
    def __init__(self, interval, *args, **kwargs):
        self.interval = interval
        super().__init__(*args, **kwargs)

    def gen_next(self):
        lb, ub = self.interval
        return np.random.uniform(lb, ub, size=(1,))


class BatchQuery(AbstractQuery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data(self, n, server):
        x = self.gen_query(n)
        y = server.respond(x)
        return x, y

    @abstractmethod
    def gen_query(self, n):
        """Abstract method to generate queries."""


class RandomQuery(BatchQuery):
    def __init__(self, interval, type='unif', *args, **kwargs):
        # self.shape = shape
        self.interval = interval
        self.type = type
        super().__init__(*args, **kwargs)

    def gen_query(self, n):
        lb, ub = self.interval
        if self.type == 'unif':
            return np.random.uniform(lb, ub, size=(n,))
        elif self.type == 'Gaussian':
            loc = self.kwargs.get('loc', 0)
            scale = self.kwargs.get('scale', 1)
            return np.clip(np.random.normal(loc, scale, size=(n,)), lb, ub)
        else:
            raise ValueError("Invalid query type.")


class EqualSpaceQuery(BatchQuery):
    def __init__(self, interval, *args, **kwargs):
        self.interval = interval
        super().__init__(*args, **kwargs)

    def gen_query(self, n):
        lb, ub = self.interval
        return np.linspace(lb, ub, n)


class SampleQuery(BatchQuery):
    def __init__(self, x, *args, **kwargs):
        # x either a numpy array or a torch.datasets
        self.x = x
        super().__init__(*args, **kwargs)

    def gen_query(self, n):
        if isinstance(self.x, np.ndarray):
            return np.random.choice(self.x, n)
        elif isinstance(self.x, Dataset):
            # return random_split(self.x, [n, len(self.x) - n,])[0]
            # return Subset(self.x, )
            idx = np.random.choice(len(self.x), n)
            tmp = [self.x[i][0][None, ] for i in idx]
            return torch.vstack(tmp)
        else:
            raise ValueError("Invalid Sample Dataset Type (only numpy.ndarray or torch.utils.data.Dataset).")


def get_query(method_name, *args, **kwargs):
    if method_name == 'IID':
        query = RandomQuery(*args, **kwargs)
    elif method_name == 'EqualSpace':
        query = EqualSpaceQuery(*args, **kwargs)
    elif method_name == 'Sequential':
        query = RandomQuery(*args, **kwargs)
    elif method_name == 'IIDSeq':
        query = RandomSequentialQuery(*args, **kwargs)
    elif method_name == 'Sample':
        query = SampleQuery(*args, **kwargs)
    else:
        raise ValueError("Invalid query strategy.")
    return query


class AbstractAlgorithm(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def train(self, x, y):
        """Abstract method to train the model."""

    @abstractmethod
    def predict(self, x):
        """Abstract method to predict the model."""


def model_selection(x, y, pmax, fit_intercept=True):
    n = len(y)
    aics = np.zeros(pmax)
    for i in range(pmax):
        X = x[:, :i+1]
        lr = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
        loss = mean_squared_error(lr.predict(X), y)
        aics[i] = loss + 2*i*loss/(n-i-1)
        logger.debug(f'Order {i}, Error {aics[i]}, Loss {loss}')
    return np.argmin(aics)


class LinearRegAIC(AbstractAlgorithm):
    def __init__(self, pmax=5, *args, **kwargs):
        self.pmax = pmax
        self.order = None
        self.fit_intercept = kwargs.get('fit_intercept', True)
        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        super().__init__(*args, **kwargs)

    def train(self, x, y):
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        basis = [legendre(i) for i in range(self.pmax)]
        X = np.hstack([basis[i](x) for i in range(self.pmax)])
        p = model_selection(X, y, self.pmax, fit_intercept=self.fit_intercept)
        self.order = p
        self.model.fit(X[:, :p+1], y)

    def predict(self, x):
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        basis = [legendre(i) for i in range(self.order+1)]
        X = np.hstack([basis[i](x) for i in range(self.order+1)])
        return self.model.predict(X)


class KernelRidgeReg(AbstractAlgorithm):
    def __init__(self, *args, **kwargs):
        self.model = KernelRidge(**kwargs)
        super().__init__(*args, **kwargs)

    def train(self, x, y):
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        self.model.fit(x, y)

    def predict(self, x):
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        return self.model.predict(x)


class NNLeNet5(AbstractAlgorithm):
    def __init__(self, model, device='cpu', batch_size=64, num_epochs=10, train_loss=None, *args, **kwargs):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_loss = train_loss
        super().__init__(*args, **kwargs)

    def train(self, x, y):
        trainset = SimData(x, y)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        scheduler = self.kwargs.get('scheduler', None)
        optimizer = self.kwargs.get('optimizer', None)
        train_loss, test_loss, model = run(trainloader, self.model,
                                           criterion=self.train_loss, criterion2=self.train_loss,
                                           test_loader=None, num_epochs=self.num_epochs, device=self.device,
                                           scheduler=scheduler, optimizer=optimizer)
        self.model = model

    def predict(self, x):
        return self.model(x.to(self.device)).detach().cpu()


def get_algorithm(method_name, *args, **kwargs):
    if method_name == 'linear':
        alg = LinearRegAIC(*args, **kwargs)
    elif method_name == 'KRR':
        alg = KernelRidgeReg(*args, **kwargs)
    elif method_name == 'LeNet5':
        alg = NNLeNet5(*args, **kwargs)
    else:
        raise ValueError("Invalid learning algorithm.")
    return alg
