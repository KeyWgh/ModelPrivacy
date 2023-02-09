"""Utility classes."""
from .defense import *
from .attack import *
from torch import Tensor, from_numpy
import logging
logger = logging.getLogger(__name__)


class Server:
    def __init__(self, model=None, device='cpu', defend_method='None', *args, **kwargs):
        self.device = device
        self.model = model
        self.defense = get_defense(defend_method, model=model, *args, **kwargs)

    def respond(self, x):
        y = self.model(x)
        flag = True if isinstance(x, Tensor) else False
        y = y.detach().numpy() if flag else y
        yhat = self.defense.respond(x, y)
        return from_numpy(yhat).type(x.dtype) if flag else yhat


class Attacker:
    def __init__(self, learning_algorithm='linear', query_strategy='IID', loss='MSE', *args
                 , alg_kwargs=None, query_kwargs=None, add_data=None):
        self.query = get_query(query_strategy, *args, **query_kwargs)
        self.algorithm = get_algorithm(learning_algorithm, *args, **alg_kwargs)
        self.data = None
        self.add_data = add_data
        self.loss_fn = mean_squared_error if loss == 'MSE' else loss

    def collect_data(self, n, server):
        self.data = self.query.get_data(n, server)

    def train(self):
        if self.add_data is not None:
            x, y = np.concatenate((self.data[0], self.add_data[0])), np.concatenate((self.data[1], self.add_data[1]))
        else:
            x, y = self.data
        self.algorithm.train(x, y)

    def evaluate(self, x, y):
        return self.loss_fn(y, self.algorithm.predict(x))
