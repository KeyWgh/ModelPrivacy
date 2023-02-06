from abc import ABC, abstractmethod
import numpy as np


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

        return x, y

    @abstractmethod
    def gen_next(self):
        """Abstract method to generate next query."""


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
    def __init__(self, shape, *args, **kwargs):
        self.shape = shape
        super().__init__(*args, **kwargs)

    def gen_query(self, n):
        return np.random.uniform(size=(n, *self.shape))


def get_query(method_name, *args, **kwargs):
    if method_name == 'IID':
        query = RandomQuery(*args, **kwargs)
    elif method_name == 'RandomNoiseClassficiation':
        query = RandomQuery(*args, **kwargs)
    else:
        raise ValueError("Invalid query strategy.")
    return query


class AbstractAlgorithm(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model = None

    @abstractmethod
    def train(self, x, y):
        """Abstract method to train the model."""

    @abstractmethod
    def predict(self, x):
        """Abstract method to predict the model."""


class LinearRegression(AbstractAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, x, y):
        self.model = np.linalg.pinv(x) @ y

    def predict(self, x):
        return x @ self.model


def get_algorithm(method_name, *args, **kwargs):
    if method_name == 'linear':
        alg = LinearRegression(*args, **kwargs)
    elif method_name == 'RandomNoiseClassficiation':
        alg = LinearRegression(*args, **kwargs)
    else:
        raise ValueError("Invalid learning algorithm.")
    return alg
