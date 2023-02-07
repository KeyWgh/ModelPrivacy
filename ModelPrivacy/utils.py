"""Utility classes."""
from .defense import *
from .attack import *
import logging
logger = logging.getLogger(__name__)


class Server:
    def __init__(self, model=None, device='cpu', defend_method='None', *args, **kwargs):
        self.device = device
        self.model = model
        self.defense = get_defense(defend_method, model=model, *args, **kwargs)

    def respond(self, x):
        return self.defense.respond(x, self.model(x))


class Attacker:
    def __init__(self, learning_algorithm='linear', query_strategy='IID', loss='MSE', *args
                 , alg_kwargs=None, query_kwargs=None):
        self.query = get_query(query_strategy, *args, **query_kwargs)
        self.algorithm = get_algorithm(learning_algorithm, *args, **alg_kwargs)
        self.data = None
        self.loss_fn = mean_squared_error if loss == 'MSE' else loss

    def collect_data(self, n, server):
        self.data = self.query.get_data(n, server)

    def train(self):
        self.algorithm.train(*self.data)

    def evaluate(self, x, y):
        return self.loss_fn(y, self.algorithm.predict(x))
