"""High-dim experiment."""
import logging
import sys, os
from pickle import dump
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from mpi4py import MPI
from ModelPrivacy import *
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LassoCV, Lasso, OrthogonalMatchingPursuit, RidgeCV, ElasticNetCV

logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def my_excepthook(excType, excValue, traceback):
    logger.error("Logging an uncaught exception",
                 exc_info=(excType, excValue, traceback))

    
def make_f(beta):
    def f(x):
        return x@beta
    return f


def make_score(beta):
    def score(clf, x, y_true):
        y_pred = clf.predict(x)
        acc = mean_squared_error(y_pred, y_true)
        y_pred = (np.abs(clf.coef_)>0)
        y_true = 1*(np.abs(beta)>0)
        f1 = f1_score(y_true, y_pred)
        diff = sum(np.abs(y_pred - y_true))
        return acc, f1, diff
    return score


def reg_iid_noise(y, sigma=1):
    return y+np.random.normal(0, sigma, size=y.size)


def reg_const_noise(y, sigma=1):
    return y+sigma


def longrange_noise(y, gamma=0.5, sigma=1, k=1):
    n = len(y)
    gamma = 1/np.power(n, k) if gamma is None else gamma
    noise = powerlaw_psd_gaussian(1-gamma, y.shape)
    return y+noise*sigma


def cal_noise(noise, y_pred, sigma):
    noise_norm = norm(noise)
    n = len(y_pred)
    if noise_norm > sigma*np.sqrt(n):
        noise = noise/noise_norm*sigma*np.sqrt(n)
    else:
        alpha = np.sqrt(n*sigma**2-noise_norm**2)
        noise = alpha*y_pred/norm(y_pred)+noise
    return noise


def lasso_noise(y, X, sigma=1, beta=None, exclude_ratio=0.3):
    beta_n = np.zeros(X.shape[1]) if beta is None else beta
    idx_avail = np.array(beta_n==0)
    if sum(idx_avail) == 0: 
        idx_avail = np.array([True]*len(beta))
    n_idx_exclude = min(sum(idx_avail)-1, int(sum(idx_avail)*exclude_ratio))
    idx_exclude = np.random.choice(idx_avail.nonzero()[0], n_idx_exclude, replace=False)
    idx_avail[idx_exclude] = False
    
    # n_coef = min((sum(idx_avail == 1)*0.8), 1)
    n_coef = sum(idx_avail)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_coef).fit(X[:, idx_avail], y)
    y_pred = omp.predict(X[:, idx_avail])
    noise = y_pred-y
    noise = cal_noise(noise, y_pred, sigma)
    return y+noise



def run(save_path='./saved_models', kind=1):
    def gen_x(kind):
        if kind == 4:
            z1 = np.random.normal(0, 1, size=(n_train+n_valid+n_test,))
            e1 = np.random.normal(0, 0.1, size=(n_train+n_valid+n_test,5))
            x1 = z1[:, None]+e1
            z2 = np.random.normal(0, 1, size=(n_train+n_valid+n_test,))
            e2 = np.random.normal(0, 0.1, size=(n_train+n_valid+n_test,5))
            x2 = z2[:, None]+e2
            z3 = np.random.normal(0, 1, size=(n_train+n_valid+n_test,))
            e3 = np.random.normal(0, 0.1, size=(n_train+n_valid+n_test,5))
            x3 = z3[:, None]+e3
            x4 = np.random.normal(0, 1, size=(n_train+n_valid+n_test,25))
            x = np.concatenate([x1, x2, x3, x4], axis=1)
        elif kind==5:
            cov = np.power(0.5, np.abs([[i-j for j in range(15) ]for i in range(15)] ))
            mean = np.zeros(15)
            x1 = np.random.multivariate_normal(mean, cov, size=(n_train+n_valid+n_test,))
            x2 = np.random.normal(0, 1, size=(n_train+n_valid+n_test, 185))
            x = np.concatenate([x1, x2], axis=1)
        elif kind == 6:
            x = np.random.normal(0, 1, size=(n_train+n_valid+n_test, 200))
            
        return x
    
    sigma = 3  # Noise level
    sigmas = np.logspace(-1, 0, 10)
    n_rep = 100   # Number of repetitions per node
    ns = [20, 50, 100, 200, 500]  # Number of training data
    
    if kind == 1:
        cov = np.power(0.5, np.abs([[i-j for j in range(8) ]for i in range(8)] ))
        beta =  np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
        n_train = 20
        n_valid = 20
        n_test = 200
        exclude_ratio = 0.3
    elif kind == 2: 
        cov = np.power(0.5, np.abs([[i-j for j in range(8) ]for i in range(8)] ))
        beta =  np.array([0.85]*8)
        n_train = 20
        n_valid = 20
        n_test = 200
        exclude_ratio = 0.3
    elif kind == 3:
        n_fea = 40
        cov = np.ones((n_fea, n_fea))*0.5
        cov += np.identity(n_fea)*0.5
        beta = np.concatenate([np.zeros(10), np.ones(10)*2, np.zeros(10), np.ones(10)*2,])
        n_train = 100
        n_valid = 100
        n_test = 400
        exclude_ratio = 0.3
    elif kind == 4:
        n_fea = 40
        beta = np.concatenate([np.ones(15)*3, np.zeros(25)])
        n_train = 50
        n_valid = 50
        n_test = 400
        exclude_ratio = 0.3
    elif kind == 5:
        n_fea = 200
        beta = np.concatenate([np.ones(5)*2.5, np.ones(5)*1.5, np.ones(5)*0.5, np.zeros(185),])
        n_train = 150
        n_valid = 50
        n_test = 400
        exclude_ratio = 0.9
    elif kind == 6:
        n_fea = 200
        beta = np.concatenate([[10, 5, 5, 2.5, 2.5, 1.25, 1.25, 5/8, 5/8, 5/16, 5/16], np.zeros(189),])
        n_train = 150
        n_valid = 50
        n_test = 400
        exclude_ratio = 0.9
    elif kind == 7:
        n_fea = 60
        beta = np.concatenate([[3, 1.5, 0, 0, 2], np.zeros(55),])
        cov = np.power(0.5, np.abs([[i-j for j in range(n_fea) ]for i in range(n_fea)] ))
        n_train = 50
        n_valid = 50
        n_test = 400
        exclude_ratio = 0.9
        
    mean = np.zeros_like(beta)
    f = make_f(beta)
    score = make_score(beta)
    
    df = pd.DataFrame(columns=['query size', 'attack', 'defense', 'un', 'error', 'f1', 'sym_diff'])
    i = 0
    sigmas = np.logspace(-1, 0, 10)
    methods = ['LASSO', 'Ridge', 'Elastic Net']
    atks = {'LASSO': LassoCV(fit_intercept=True),
           'Ridge': RidgeCV(fit_intercept=True),
           'Elastic Net': ElasticNetCV(fit_intercept=True),
           }
    for num_query in [20]:
        for m in methods:
            atk = atks[m]
            logger.warning(f'Query size {num_query}, attack {m}')
            for k in tqdm(range(20)): 
                x = gen_x(kind) if kind in [4, 5, 6] else np.random.multivariate_normal(mean, cov, size=(n_train+n_valid+n_test,))
                y = f(x)
                x_train, y_train = x[:n_train], y[:n_train]
                x_val, y_val = x[n_train:n_train+n_valid], y[n_train:n_train+n_valid]
                x_test, y_test = x[n_train+n_valid:n_train+n_valid+n_test], y[n_train+n_valid:n_train+n_valid+n_test] 

                clf = atk.fit(x_train, y_train)
                res = score(clf, x_test, y_test)
                df.loc[i] = [num_query, m, 'No', 0, *res]
                i += 1

                for an in sigmas:
                    clf = atk.fit(x_train, reg_iid_noise(y_train, sigma=an))
                    res = score(clf, x_test, y_test)
                    # df.loc[i] = [num_query, 'IID', mean_squared_error(y_val, reg_iid_noise(y_val, sigma=an)), *res]
                    df.loc[i] = [num_query, m, 'IID', an**2, *res]
                    i += 1

                for an in sigmas: 
                    clf = atk.fit(x_train, reg_const_noise(y_train, sigma=an))
                    res = score(clf, x_test, y_test)
                    # df.loc[i] = [num_query, 'Const', mean_squared_error(y_val, reg_const_noise(y_val, sigma=an)), *res]
                    df.loc[i] = [num_query, m, 'Const', an**2, *res]
                    i += 1

                for an in sigmas: 
                    clf = atk.fit(x_train, longrange_noise(y_train, gamma=None, sigma=an))
                    res = score(clf, x_test, y_test)
                    # df.loc[i] = [num_query, 'Corr', mean_squared_error(y_val, longrange_noise(y_val, gamma=an)), *res]
                    df.loc[i] = [num_query, m, 'Corr', an**2, *res]
                    i += 1

                for an in sigmas:
                    clf = atk.fit(x_train, lasso_noise(y_train, x_train, sigma=an, beta=beta, exclude_ratio=exclude_ratio))
                    res = score(clf, x_test, y_test)
                    # df.loc[i] = [num_query, 'LASSO', mean_squared_error(y_val, lasso_noise(y_val, x_val, sigma=an, beta=beta)), *res]
                    df.loc[i] = [num_query, m, 'LASSO', an**2, *res]
                    i += 1
    
    df.to_csv(f'./saved_models/lasso_sim_{kind}.csv')


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
    
    for kind in [1, 2, 3, 4]:
        logger.warning(f'kind = {kind}')
        run(save_path=save_path, kind=kind)
