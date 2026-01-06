"""Utility classes."""
import numpy as np

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
        y = self.model(x).squeeze()
        flag = True if isinstance(x, Tensor) else False
        y = y.detach().cpu().numpy() if flag else y
        yhat = self.defense.respond(x, y)
        return from_numpy(yhat).type(x.dtype) if flag else yhat


class Attacker:
    def __init__(self, learning_algorithm='linear', query_strategy='IID', loss='MSE', *args,
                 alg_kwargs=None, query_kwargs=None, add_data=None):
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


class Validator:
    def __init__(self, loss, x=None, y=None,):
        self.loss_fn = loss
        self.x = x
        self.y = y

    def eval(self, model):
        yhat = model.predict(self.x)
        return self.loss_fn(yhat, self.y)


def paste(x, y, decimal=2):
    """Utility function to print `x (y)` with specified decimal digits."""
    def _paste(x, y, decimal=2):
        return f'{np.round(x, decimals=decimal)} ({np.round(y, decimals=decimal)})'

    func = np.vectorize(_paste)
    return func(x, y, decimal=decimal)


def clf_iid_noise(y, p=0.1):
    l = len(y)
    # Scale the probability to have sum one
    d = y.shape[1]-1 if len(y.shape) > 1 else 1
    # Sample flip indices
    flip = np.random.choice(range(l), int(l*p), replace=False)
    # make copy
    yhat = 1*y
    # flip
    # yhat[flip] = (1-yhat[flip])/d
    x = yhat[flip]
    indices = torch.argsort(torch.rand_like(x), dim=-1)
    result = torch.gather(x, dim=-1, index=indices)
    yhat[flip] = result
    return yhat


def clf_const(y, p=0.1):
    if len(y.shape) > 1:
        d = y.shape[1]
        ind = y.argmax(dim=1).mode()[0].item()
        sign = torch.zeros(d)
        sign[ind] = p
    else:
        sign = 1 if (sum(y > 0.5) > len(y) / 2) else -1
        sign *= p
    return F.softmax(torch.log(y)+sign, dim=1)


def clf_am(y, p=0.1):
    d = y.shape[1] if len(y.shape) > 1 else 2
    flip = 1*(y.max(dim=1)[0]<1/d+p)[:,None]
    rand = torch.rand_like(y)
    rand = rand/(rand.sum(dim=1)[:, None])
    return y*(1-flip)+rand*flip


def clf_dp(y, beta=0.5, gamma=0.2):
    # flip = 1*(torch.abs(y-0.5)>0.5-p)
    # d = y.shape[1] if len(y.shape) > 1 else 2
    r = beta*(torch.sigmoid(gamma*torch.logit(y))-0.5)
    # rand = torch.rand_like(y)
    yhat = y-r
    return yhat/(yhat.sum(dim=1)[:, None])


def reg_iid_noise(y, sigma=1):
    return y+np.random.normal(0, 1, size=y.size)


def longrang_noise(y, gamma=0.5, sigma=1):
    noise = powerlaw_psd_gaussian(1-gamma, y.shape)
    return y+noise*sigma


def my_excepthook(excType, excValue, traceback):
    logger.error("Logging an uncaught exception",
                 exc_info=(excType, excValue, traceback))
