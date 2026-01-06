import matplotlib.pyplot as plt
from datasets import load_dataset_builder, get_dataset_split_names, load_dataset, Dataset, load_from_disk

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, roc_auc_score
import random
import pickle
import sys
import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
module_path = './projects/MP'
if module_path not in sys.path:
    sys.path.append(module_path)

from util import *
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


class BaseClassifier(nn.Module):
    def __init__(self, val):
        super(BaseClassifier, self).__init__()
        self.val = val

    def forward(self, x):
        return self.val*torch.ones(x.shape[0])


def score(clf, x, y_true, device='cpu'):
    y_pred = clf(x.to(device)).detach().cpu().squeeze()
    acc = 1-ZeroOneLoss(y_pred, y_true).item()
    f1 = f1_score(y_true, y_pred>0.5)
    auc = roc_auc_score(y_true, y_pred)
    return acc, f1, auc


def main(model_balance, test_balance, balance):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using {device}')
    tag = ''
    reg = ''
    # model_balance = ''
    # test_balance = ''
    # balance = ''

    teacher_dataset_name = 'ToxicCommentChallenge'  # hate_speech_offensive
    df = pd.DataFrame(columns=['query dataset', 'query size', 'defense', 'un', 'acc', 'f1', 'auc', 'train size'])
    i = 0

    for query_dataset_name in ['ToxicCommentChallenge', 'hate_speech_offensive']:
        teacher_model_name = 'bert-base-uncased'  # 'bert-base-uncased'
        attacker_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  #
        teacher_CLS = False
        attacker_CLS = False

        batch_size = 320

        teacher_filename = f'embed_{teacher_model_name.replace("/", "_")}_{"CLS" if teacher_CLS else "Avg"}'
        teacher_model_path = f"pretrained/{teacher_dataset_name}/{teacher_filename.replace('/', '_')}_{tag}{model_balance}.pth"
        attacker_filename = f'embed_{attacker_model_name.replace("/", "_")}_{"CLS" if teacher_CLS else "Avg"}'

        # Attacker's query embedding
        query_teacher_embedding_path = f'./benchmark/{query_dataset_name}/{teacher_filename}.pkl'
        query_attacker_embedding_path = f'./benchmark/{query_dataset_name}/{attacker_filename}.pkl'

        with open(query_teacher_embedding_path, "rb") as input_file:
            X_teacher = pickle.load(input_file)['feature'].to_numpy()

        with open(query_attacker_embedding_path, "rb") as input_file:
            X_attacker = pickle.load(input_file)['feature'].to_numpy()

        # Attacker's test embedding
        attacker_test_embed_path = f'./benchmark/{teacher_dataset_name}/{attacker_filename}.pkl'
        with open(attacker_test_embed_path, "rb") as input_file:
            X_atk_test = pickle.load(input_file)['feature'].to_numpy()

        train_embed_path = f'./benchmark/{teacher_dataset_name}/{teacher_filename}.pkl'
        with open(train_embed_path, "rb") as input_file:
            X = pickle.load(input_file)['feature'].to_numpy()

        # Each raw of X is a vector
        X_teacher = np.vstack(X_teacher)
        X_attacker = np.vstack(X_attacker)
        X = np.vstack(X)
        X_atk_test = np.vstack(X_atk_test)

        # read original dataset
        excel_file_path = f"rawdata/{teacher_dataset_name}.csv"
        data = pd.read_csv(excel_file_path)
        y = data['normal'].to_numpy()

        # Balance the data by randomly sampling from the larger set
        if test_balance:
            min_data_size = min(sum(y == 0), sum(y == 1))
            indices_hate = np.nonzero(y == 0)[0]
            np.random.shuffle(indices_hate)
            indices_normal = np.nonzero(y == 1)[0]
            np.random.shuffle(indices_normal)
            indices = np.concatenate([indices_hate[:min_data_size], indices_normal[:min_data_size]])
            X, y = X[indices, :], y[indices]
            X_atk_test = X_atk_test[indices, :]

        n_tot = len(y)
        train_size = int(0.4 * n_tot)
        val_size = int(0.3 * n_tot)
        test_size = n_tot - train_size - val_size

        ind = shuffle(list(range(n_tot)))
        train_ind = ind[:train_size]
        val_ind = ind[train_size:train_size + val_size]
        test_ind = ind[train_size + val_size:]

        train_dataset = TensorDataset(torch.tensor(X[train_ind]), torch.tensor(y[train_ind]))
        val_dataset = TensorDataset(torch.tensor(X[val_ind]), torch.tensor(y[val_ind]))
        test_dataset = TensorDataset(torch.tensor(X[test_ind]), torch.tensor(y[test_ind]))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        X_test = torch.tensor(X_atk_test[test_ind])
        y_test = torch.tensor(y[test_ind])
        atk_test_dataset = TensorDataset(X_test, y_test)
        atk_test_loader = DataLoader(atk_test_dataset, batch_size=batch_size, shuffle=False)
        test_crit = ZeroOneLoss

        # Load teacher
        teacher = BinaryClassifier(X_teacher.shape[1], 256).to(device)
        teacher.load_state_dict(torch.load(teacher_model_path, map_location=device))
        # logger.info(f'Teacher test error: {test(teacher, test_loader, device=device, criterion=test_crit):.4f}')
        y_val = teacher((val_dataset.tensors[0][:3000]).to(device)).detach().cpu().squeeze()

        val = 1 if (sum(y == 1) > sum(y == 0)) else 0
        res = score(BaseClassifier(val), X_test, y_test, device=device)
        df.loc[i] = [query_dataset_name, 0, 'Any', 0, *res, 0]
        i += 1

        for num_query in [1000, 2000, 5000]:
            logger.info(f'Query from {query_dataset_name}, {num_query}')

            query_ind = np.random.choice(len(X_teacher), num_query, replace=False)
            # Teacher response
            y_response = teacher(torch.Tensor(X_teacher[query_ind, :]).to(device)).detach().cpu().squeeze()
            X = torch.tensor(X_attacker[query_ind])
            y = y_response
            num_epochs = 10
            criterion = nn.BCELoss()

            if balance:
                y = y.numpy()
                min_data_size = min(sum(y < 0.5), sum(y > 0.5))
                indices_hate = np.nonzero(y < 0.5)[0]
                np.random.shuffle(indices_hate)
                indices_normal = np.nonzero(y > 0.5)[0]
                np.random.shuffle(indices_normal)
                indices = np.concatenate([indices_hate[:min_data_size], indices_normal[:min_data_size]])
                X, y = X[indices, :], torch.Tensor(y[indices])

            atk_train_size = len(y)

            for k in tqdm(range(10)):
                train_dataset = TensorDataset(X, y)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                classifier = train(BinaryClassifier(X.shape[1], 256).to(device), train_loader, verbose=False,
                                   criterion=criterion, device=device,
                                   num_epochs=num_epochs, model_path=None)
                res = score(classifier, X_test, y_test, device=device)
                df.loc[i] = [query_dataset_name, num_query, 'No', 0, *res, atk_train_size]
                i += 1

                for an in [0.01, 0.02, 0.05, 0.1, 0.15]:
                    train_dataset = TensorDataset(X, clf_iid_noise(y, p=an))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    classifier = train(BinaryClassifier(X.shape[1], 256).to(device), train_loader, verbose=False,
                                       criterion=criterion, device=device,
                                       num_epochs=num_epochs, model_path=None)
                    res = score(classifier, X_test, y_test, device=device)
                    df.loc[i] = [query_dataset_name, num_query, 'IID', test_crit(y_val, clf_iid_noise(y_val, p=an)).item(), *res, atk_train_size]
                    i += 1

                for p in [0.1, 0.2, 0.3, 0.4, 0.5]: 
                    train_dataset = TensorDataset(X, clf_dp(y, beta=p))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    classifier = train(BinaryClassifier(X.shape[1], 256).to(device), train_loader, verbose=False,
                                       criterion=criterion, device=device,
                                       num_epochs=num_epochs, model_path=None)
                    res = score(classifier, X_test, y_test, device=device)
                    df.loc[i] = [query_dataset_name, num_query, 'DP', test_crit(y_val, clf_dp(y_val, beta=p)).item(), *res, atk_train_size]
                    i += 1

                for p in [0.04, 0.09, 0.2, 0.33, 0.41,]: 
                    train_dataset = TensorDataset(X, clf_am(y, p=p))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    classifier = train(BinaryClassifier(X.shape[1], 256).to(device), train_loader, verbose=False,
                                       criterion=criterion, device=device,
                                       num_epochs=num_epochs, model_path=None)
                    res = score(classifier, X_test, y_test, device=device)
                    df.loc[i] = [query_dataset_name, num_query, 'AM', test_crit(y_val, clf_am(y_val, p=p)).item(), *res, atk_train_size]
                    i += 1

                for p in [0.18, 0.35, 0.84, 1.65, 2.33]:  
                    train_dataset = TensorDataset(X, clf_const(y, p=p))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    classifier = train(BinaryClassifier(X.shape[1], 256).to(device), train_loader, verbose=False,
                                       criterion=criterion, device=device,
                                       num_epochs=num_epochs, model_path=None)
                    res = score(classifier, X_test, y_test, device=device)
                    df.loc[i] = [query_dataset_name, num_query, 'Const', test_crit(y_val, clf_const(y_val, p=p)).item(), *res, atk_train_size]
                    i += 1

    df.to_csv(f'./result/steal_{teacher_dataset_name}_{teacher_model_name.replace("/", "_")}{model_balance}_test{test_balance}_atk{balance}.csv', index=False)


if __name__ == "__main__":
    logPath = './logs/'
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    sys.excepthook = my_excepthook
    fileName = str(datetime.now()).split('.')[0]
    logging.basicConfig(
        # level=logging.DEBUG,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("{0}{1}.log".format(logPath, fileName)),
            # logging.StreamHandler()
        ]
    )
    for model_balance in ['_balance']:
        for test_balance in ['_balance']:
            for balance in ['_balance']:
                main(model_balance, test_balance, balance)
