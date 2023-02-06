"""Utility classes."""
from defense import *
from attack import *


class Server:
    def __init__(self, model=None, device='cpu', defend_method='None', *args, **kwargs):
        self.device = device
        self.model = model
        self.defense = get_defense(defend_method, model=model, *args, **kwargs)

    def respond(self, x):
        return self.defense.respond(x, self.model(x))


class Attacker:
    def __init__(self, learning_algorithm, query_strategy, device='cpu', *args, **kwargs):
        self.device = device
        self.query = get_query(query_strategy, *args, **kwargs)
        self.algorithm = get_algorithm(learning_algorithm, *args, **kwargs)
        self.data = None

    def collect_data(self, n, server):
        self.data = self.query.get_data(n, server)

    def train(self):
        self.algorithm.train(*self.data)

