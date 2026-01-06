import matplotlib.pyplot as plt
# from datasets import load_dataset_builder, get_dataset_split_names, load_dataset, Dataset, load_from_disk
from collections import defaultdict
import torchvision
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
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
module_path = './projects/ModelPrivacy'
if module_path not in sys.path:
    sys.path.append(module_path)

# from util import *
from ModelPrivacy import *
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


class BaseClassifier(nn.Module):
    def __init__(self, cat):
        super(BaseClassifier, self).__init__()
        self.cat = cat

    def forward(self, x):
        prob = torch.rand(x.shape[0], self.cat)
        prob = prob/(prob.sum(dim=1)[:, None])
        return prob


def score(clf, x, y_true, device='cpu'):
    y_pred = clf(x.to(device)).detach().cpu().squeeze()
    # acc = 1-zero_one_loss(y_pred, y_true).item()
    # f1 = f1_score(y_true, y_pred.argmax(dim=1) if y_pred.ndim > 1 else y_pred, average='weighted')
    # auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
    # return acc, f1, auc
    
    acc = zero_one_loss(y_pred, y_true).item()
    ce = CELoss(y_pred, y_true).item()
    return acc, ce


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using {device}')

    path = '~/projects'
    teacher_dataset_name = 'FashionMNIST'  # hate_speech_offensive
    transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()])

    # df = pd.DataFrame(columns=['query dataset', 'query size', 'defense', 'un', 'acc', 'f1', 'auc', 'train size'])
    df = pd.DataFrame(columns=['query dataset', 'query size', 'defense', 'un', 'un_ce', 'acc', 'mp_ce', 'train size'])
    i = 0

    for query_dataset_name in ['FashionMNIST', 'MNIST']:
        teacher_model_name = 'LeNet'  # 'bert-base-uncased'
        attacker_model_name = 'LeNet'  #
        batch_size = 64

        model_path = './saved_models'
        teacher_filename = f'fmnist_teacher'
        teacher_model_path = f"{model_path}/{teacher_filename}.pt"
        attacker_filename = f''

        data_teacher_train = datasets.FashionMNIST(root=f'{path}/Datasets', train=True, download=False,
                                                   transform=transform)
        data_teacher_test = datasets.FashionMNIST(root=f'{path}/Datasets', train=False, download=False,
                                                  transform=transform)
        data_attacker = eval(
            f"datasets.{query_dataset_name}(root=f'{path}/Datasets', train=True, download=False, transform=transform)")

        X_atk_test = np.vstack([x[0][0].numpy()[None, None, :] for x in data_teacher_test])
        X = np.vstack([x[0][0].numpy()[None, None, :] for x in data_teacher_train])
        X_attacker = np.vstack([x[0][0].numpy()[None, None, :] for x in data_attacker])
        X_teacher = X_attacker
        y = data_teacher_train.targets.numpy()

        X_test = torch.tensor(X_atk_test)
        y_test = data_teacher_test.targets

        n_tot = len(y)
        train_size = int(0.7 * n_tot)
        val_size = int(0.3 * n_tot)

        ind = shuffle(list(range(n_tot)))
        train_ind = ind[:train_size]
        val_ind = ind[train_size:]

        train_dataset = TensorDataset(torch.tensor(X[train_ind]), torch.tensor(y[train_ind]))
        val_dataset = TensorDataset(torch.tensor(X[val_ind]), torch.tensor(y[val_ind]))
        # test_dataset = TensorDataset(torch.tensor(X[test_ind]), torch.tensor(y[test_ind]))
        atk_test_dataset = data_teacher_test

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        atk_test_loader = DataLoader(atk_test_dataset, batch_size=batch_size, shuffle=False)
        test_loader = atk_test_loader
        test_crit = zero_one_loss

        # Load teacher
        teacher = LeNet5(10).to(device)
        teacher.load_state_dict(torch.load(teacher_model_path, map_location=device))
        # logger.info(f'Teacher test error: {test(teacher, test_loader, device=device, criterion=test_crit):.4f}')
        y_val = teacher((val_dataset.tensors[0][:3000]).to(device)).detach().cpu().squeeze()

        # if evaluate model privacy instead of accuracy.
        y_test = teacher(X_test.to(device)).detach().cpu().squeeze()
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        res = score(BaseClassifier(10), X_test, y_test, device=device)
        df.loc[i] = [query_dataset_name, 0, 'Any', 0, 0, *res, 0]
        i += 1
        

        for num_query in [1000, 2000, 5000]:
            logger.info(f'Query from {query_dataset_name}, {num_query}')

            query_ind = np.random.choice(len(X_teacher), num_query, replace=False)
            # Teacher response
            y_response = teacher(torch.Tensor(X_teacher[query_ind, :]).to(device)).detach().cpu().squeeze()
            X = torch.tensor(X_attacker[query_ind])
            y = y_response
            num_epochs = 50
            criterion = CELoss

            atk_train_size = len(y)
 
            for k in tqdm(range(10)):
                train_dataset = TensorDataset(X, y)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                classifier = train(LeNet5(10).to(device), train_loader, verbose=False,
                                   criterion=criterion, device=device,
                                   num_epochs=num_epochs, model_path=None)
                res = score(classifier, X_test, y_test, device=device)
                df.loc[i] = [query_dataset_name, num_query, 'No', 0, 0, *res, atk_train_size]
                i += 1

                for an in [0.01, 0.02, 0.05, 0.1, 0.15]:
                    train_dataset = TensorDataset(X, clf_iid_noise(y, p=an))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    classifier = train(LeNet5(10).to(device), train_loader, verbose=False,
                                       criterion=criterion, device=device,
                                       num_epochs=num_epochs, model_path=None)
                    res = score(classifier, X_test, y_test, device=device)
                    df.loc[i] = [query_dataset_name, num_query, 'IID', test_crit(y_val, clf_iid_noise(y_val, p=an)).item(), 
                                 CELoss(clf_iid_noise(y_val, p=an), y_val).item(), *res, atk_train_size]
                    i += 1

                for p in [0.1, 0.2, 0.3, 0.4, 0.5]: 
                    train_dataset = TensorDataset(X, clf_dp(y, beta=p))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    classifier = train(LeNet5(10).to(device), train_loader, verbose=False,
                                       criterion=criterion, device=device,
                                       num_epochs=num_epochs, model_path=None)
                    res = score(classifier, X_test, y_test, device=device)
                    df.loc[i] = [query_dataset_name, num_query, 'DP', test_crit(y_val, clf_dp(y_val, beta=p)).item(), 
                                 CELoss(clf_dp(y_val, beta=p),y_val).item(), *res, atk_train_size]
                    i += 1

                for p in [0.4, 0.5, 0.6, 0.7, 0.75]:  
                    train_dataset = TensorDataset(X, clf_am(y, p=p))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    classifier = train(LeNet5(10).to(device), train_loader, verbose=False,
                                       criterion=criterion, device=device,
                                       num_epochs=num_epochs, model_path=None)
                    res = score(classifier, X_test, y_test, device=device)
                    df.loc[i] = [query_dataset_name, num_query, 'AM', test_crit(y_val, clf_am(y_val, p=p)).item(), 
                                 CELoss(clf_am(y_val, p=p), y_val).item(), *res, atk_train_size]
                    i += 1

                for p in [2, 5, 6, 7, 8]: 
                    train_dataset = TensorDataset(X, clf_const(y, p=p))
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    classifier = train(LeNet5(10).to(device), train_loader, verbose=False,
                                       criterion=criterion, device=device,
                                       num_epochs=num_epochs, model_path=None)
                    res = score(classifier, X_test, y_test, device=device)
                    df.loc[i] = [query_dataset_name, num_query, 'Const', test_crit(y_val, clf_const(y_val, p=p)).item(), 
                                 CELoss(clf_const(y_val, p=p), y_val).item(), *res, atk_train_size]
                    i += 1

    df.to_csv(f'./saved_models/steal_{teacher_dataset_name}_{teacher_model_name.replace("/", "_")}_mp.csv', index=False)


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
    main()
