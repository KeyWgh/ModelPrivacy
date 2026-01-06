"""Polynomial experiment."""
import logging
import sys, os
from pickle import dump
from datetime import datetime

import numpy as np
from mpi4py import MPI
from ModelPrivacy import *
logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def my_excepthook(excType, excValue, traceback):
    logger.error("Logging an uncaught exception",
                 exc_info=(excType, excValue, traceback))


def exp(server, attacker, n_train=100, valid=None):
    # Collect data and train the attacker
    attacker.collect_data(n_train, server)
    attacker.train()
    # Evaluate the attacker's test error
    return valid.eval(attacker.algorithm)


def run(save_path='./saved_models', penalty='AIC'):
    def f(x):
        # defender's function
        return (x - 0.5) ** 2 * 4

    sigma = 0.5  # Noise level
    sigmas = np.logspace(-1, 0, 10)
    n_rep = 100 // size  # Number of repetitions per node
    ns = [20, 50, 100, 200, 500]  # Number of training data
    n_train = 100
    n_test = 1000

    defenses = ['None', 'IID', 'Const', 'Corr', 'Poly', 'CorrSort']

    for k, sigma in enumerate(sigmas):
        defense_dict = {
            'None': dict(defend_method='None'),
            'IID': dict(defend_method='RandomNoiseRegression', sigma=sigma),
            'Const': dict(defend_method='UniformShift', sigma=sigma),
            'Corr': dict(defend_method='LongRangeNoise', sigma=sigma, gamma=None, sort=False),
            'CorrSort': dict(defend_method='LongRangeNoise', sigma=sigma, gamma=0.5, sort=True),
            'Poly': dict(defend_method='LR', sigma=sigma, p=int(np.power(n_train, 1/3)))
        }
        x_test = np.random.beta(1, 3, size=(n_test,))
        # x_test = np.random.uniform(0, 1, size=(n_test,))
        # x_test = np.random.normal(0, 1, size=(n_test,))
        y_test = f(x_test)
        valid = Validator(mean_squared_error, x_test, y_test)
        for j, defense in enumerate(defenses):
            test_error = np.zeros(n_rep)
            an = np.zeros(n_rep)
            alg_dict = dict(learning_algorithm='linear', alg_kwargs=dict(pmax=int(np.power(n_train, 1/3)),
                                                                         fit_intercept=False, penalty=penalty))
            query_dict = dict(query_strategy='IID', query_kwargs=dict(interval=[0, 1], type='Beta', alpha=1, beta=3))
            # query_dict = dict(query_strategy='IID', query_kwargs=dict(interval=[0, 1], type='unif'))
            # query_dict = dict(query_strategy='IID', query_kwargs=dict(interval=[-np.inf, np.inf], type='Gaussian', loc=0, scale=1))
            # Initialize the defender and attacker
            server = Server(model=f, **defense_dict[defense])
            attacker = Attacker(**query_dict, **alg_dict)
            for i in range(n_rep):
                test_error[i] = exp(server, attacker, n_train, valid)
                an[i] = sigma**2

            test_error = comm.gather(test_error, root=0)
            an = comm.gather(an, root=0)
            if rank == 0:
                logger.warning('sigma: {}, Method: {}'.format(sigma, defense))
                res = np.concatenate(test_error, axis=0)
                ans = np.concatenate(an, axis=0)
                with open(f'{save_path}/sigma_{sigma}_{defense}_{penalty}.pkl', 'wb') as output:
                    dump(dict(err=res, an=ans), output)
    
    sigma = 0.5
    for k, n_train in enumerate(ns):
        defense_dict = {
            'None': dict(defend_method='None'),
            'IID': dict(defend_method='RandomNoiseRegression', sigma=sigma),
            'Const': dict(defend_method='UniformShift', sigma=sigma),
            'Corr': dict(defend_method='LongRangeNoise', sigma=sigma, gamma=None, sort=False),
            'CorrSort': dict(defend_method='LongRangeNoise', sigma=sigma, gamma=0.5, sort=True),
            'Poly': dict(defend_method='LR', sigma=sigma, p=int(np.power(n_train, 1/3)))
        }
        x_test = np.random.beta(1, 3, size=(n_test,))
        # x_test = np.random.uniform(0, 1, size=(n_test,))
        # x_test = np.random.normal(0, 1, size=(n_test,))
        y_test = f(x_test)
        valid = Validator(mean_squared_error, x_test, y_test)
        for j, defense in enumerate(defenses):
            test_error = np.zeros(n_rep)
            an = np.zeros(n_rep)
            alg_dict = dict(learning_algorithm='linear', alg_kwargs=dict(pmax=int(np.power(n_train, 1/3)),
                                                                         fit_intercept=False, penalty=penalty))
            query_dict = dict(query_strategy='IID', query_kwargs=dict(interval=[0, 1], type='Beta', alpha=1, beta=3))
            # query_dict = dict(query_strategy='IID', query_kwargs=dict(interval=[0, 1], type='unif'))
            # query_dict = dict(query_strategy='IID', query_kwargs=dict(interval=[-np.inf, np.inf], type='Gaussian', loc=0, scale=1))
            # Initialize the defender and attacker
            server = Server(model=f, **defense_dict[defense])
            attacker = Attacker(**query_dict, **alg_dict)
            for i in range(n_rep):
                test_error[i] = exp(server, attacker, n_train, valid)
                an[i] = sigma**2

            test_error = comm.gather(test_error, root=0)
            an = comm.gather(an, root=0)
            if rank == 0:
                logger.warning('sigma: {}, Method: {}'.format(sigma, defense))
                res = np.concatenate(test_error, axis=0)
                ans = np.concatenate(an, axis=0)
                with open(f'{save_path}/n_{n_train}_{defense}_{penalty}.pkl', 'wb') as output:
                    dump(dict(err=res, an=ans), output)


if __name__ == '__main__':
    logPath = './logs/'
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    sys.excepthook = my_excepthook
    fileName = str(datetime.now()).split('.')[0]
    logging.basicConfig(
        # level=logging.INFO,
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("{0}{1}.log".format(logPath, fileName)),
            # logging.StreamHandler()
        ]
    )
    save_path = './saved_models/poly'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for penalty in ['AIC']:  # ['AIC', 'BIC', 'BC']
        run(save_path=save_path, penalty=penalty)
