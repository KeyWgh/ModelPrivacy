"""KRR experiment."""
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


def run(save_path='./saved_models'):
    def f(x):
        """Target function."""
        return x - 1.2 * x ** 2 - 0.8 * x ** 3 + 0.6 * np.cos(2 * np.pi * x)

    sigma = 0.5  # Noise level
    sigmas = np.logspace(-1, 0, 10)
    n_rep = 20 // size  # Number of repetitions per node
    ns = [20, 50, 100, 200, 500]  # Number of training data
    lb = -1
    # Initialize the validator
    n_test = 1000
    x_test = np.random.uniform(lb, 1, size=(n_test,))
    y_test = f(x_test)
    valid = Validator(mean_squared_error, x_test, y_test)

    defenses = ['None', 'IID', 'Const', 'Corr', 'KRR'] 
    krr = dict(defend_method='KRR', sigma=sigma)
    krr.update(dict(query_strategy='IID', query_kwargs=dict(interval=[lb, 1], type='unif')))
    krr.update(dict(alpha=1e-2, kernel='poly', kernel_params=dict()))

    for k, n_train in enumerate(ns):
        defense_dict = {
            'None': dict(defend_method='None'),
            'IID': dict(defend_method='RandomNoiseRegression', sigma=sigma),
            #              'CorrSort': dict(defend_method='LongRangeNoise', sigma=sigma, gamma=0.5),
            'Const': dict(defend_method='UniformShift', sigma=sigma),
            'Corr': dict(defend_method='LongRangeNoise', sigma=sigma, gamma=None, sort=False),
            'KRR': krr,
            'Poly': dict(defend_method='LR', sigma=sigma, p=5)
        }
        for j, defense in enumerate(defenses):
            test_error = np.zeros(n_rep)
            an = np.zeros(n_rep)
            alg_dict = dict(learning_algorithm='KRR',
                            alg_kwargs=dict(alpha=1e-3, kernel='rbf', ))
            query_dict = dict(query_strategy='IID', query_kwargs=dict(interval=[lb, 1], type='unif'))
            # Initialize the defender and attacker
            server = Server(model=f, **defense_dict[defense])
            attacker = Attacker(**query_dict, **alg_dict)
            for i in range(n_rep):
                # if (defense == 'KRR') and (rank == 0):
                #     logger.warning(f'Iter: {i}')
                test_error[i] = exp(server, attacker, n_train, valid)
                an[i] = f(attacker.data[0]).mean()/sigma

            test_error = comm.gather(test_error, root=0)
            an = comm.gather(an, root=0)
            if rank == 0:
                logger.warning('n: {}, Method: {}'.format(n_train, defense))
                res = np.concatenate(test_error, axis=0)
                # if (defense == 'KRR'):
                #     logger.warning(f'{res.mean()}')
                ans = np.concatenate(an, axis=0)
                with open(f'{save_path}/n_{n_train}_{defense}.pkl', 'wb') as output:
                    dump(dict(err=res, an=ans), output)
    
    n_train = 100
    for k, sigma in enumerate(sigmas):
        krr['sigma'] = sigma
        defense_dict = {
            'None': dict(defend_method='None'),
            'IID': dict(defend_method='RandomNoiseRegression', sigma=sigma),
            #              'CorrSort': dict(defend_method='LongRangeNoise', sigma=sigma, gamma=0.5),
            'Const': dict(defend_method='UniformShift', sigma=sigma),
            'Corr': dict(defend_method='LongRangeNoise', sigma=sigma, gamma=None, sort=False),
            'KRR': krr,
            'Poly': dict(defend_method='LR', sigma=sigma, p=5)
        }
        for j, defense in enumerate(defenses):
            test_error = np.zeros(n_rep)
            an = np.zeros(n_rep)
            alg_dict = dict(learning_algorithm='KRR',
                            alg_kwargs=dict(alpha=1e-3, kernel='rbf', ))
            query_dict = dict(query_strategy='IID', query_kwargs=dict(interval=[lb, 1], type='unif'))
            # Initialize the defender and attacker
            server = Server(model=f, **defense_dict[defense])
            attacker = Attacker(**query_dict, **alg_dict)
            for i in range(n_rep):
                # if (defense == 'KRR') and (rank == 0):
                #     logger.warning(f'Iter: {i}')
                test_error[i] = exp(server, attacker, n_train, valid)
                an[i] = f(attacker.data[0]).mean()/sigma

            test_error = comm.gather(test_error, root=0)
            an = comm.gather(an, root=0)
            if rank == 0:
                logger.warning('n: {}, Method: {}'.format(n_train, defense))
                res = np.concatenate(test_error, axis=0)
                # if (defense == 'KRR'):
                #     logger.warning(f'{res.mean()}')
                ans = np.concatenate(an, axis=0)
                with open(f'{save_path}/sigma_{sigma}_{defense}.pkl', 'wb') as output:
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
    save_path = './saved_models/krr'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    run(save_path=save_path)
