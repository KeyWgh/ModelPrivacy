from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from scipy.special import legendre
import logging
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
    def __init__(self, interval, *args, **kwargs):
        # self.shape = shape
        self.interval = interval
        super().__init__(*args, **kwargs)

    def gen_query(self, n):
        lb, ub = self.interval
        return np.random.uniform(lb, ub, size=(n,))


class EqualSpaceQuery(BatchQuery):
    def __init__(self, interval, *args, **kwargs):
        self.interval = interval
        super().__init__(*args, **kwargs)

    def gen_query(self, n):
        lb, ub = self.interval
        return np.linspace(lb, ub, n)


def get_query(method_name, *args, **kwargs):
    if method_name == 'IID':
        query = RandomQuery(*args, **kwargs)
    elif method_name == 'EqualSpace':
        query = EqualSpaceQuery(*args, **kwargs)
    elif method_name == 'Sequential':
        query = RandomQuery(*args, **kwargs)
    elif method_name == 'IIDSeq':
        query = RandomSequentialQuery(*args, **kwargs)
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
        aics[i] = mean_squared_error(lr.predict(X), y) + 2*i/n
        logger.debug(f'Order {i}, Error {mean_squared_error(lr.predict(X), y)}, Penalty {2*i/n}')
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


def get_algorithm(method_name, *args, **kwargs):
    if method_name == 'linear':
        alg = LinearRegAIC(*args, **kwargs)
    elif method_name == 'KRR':
        alg = KernelRidgeReg(*args, **kwargs)
    else:
        raise ValueError("Invalid learning algorithm.")
    return alg
