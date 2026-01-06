from datasets import load_dataset_builder, get_dataset_split_names, load_dataset, Dataset, load_from_disk

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, roc_auc_score
import random
import pickle
import sys
import os
import argparse
module_path = '/panfs/jay/groups/3/dingj/wang9019/projects/MP'
if module_path not in sys.path:
    sys.path.append(module_path)

from util import *
import numpy as np
import pandas as pd
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Computing on {device}')
    support_CLS = False
    batch_size = 320
    df = pd.DataFrame(columns=['dataset', 'model arch', 'tag', 'training size', 'class ratio', 'accuracy', 'f1', 'auc'])
    i = 0
    for dataset_name, tag in [('ToxicCommentChallenge', ''), ('ToxicCommentChallenge', 'toxic'),
                              ('hate_speech_offensive', ''), ('hate_speech_offensive', 'offensive')]:
        for model_name in ['sentence-transformers/all-MiniLM-L6-v2', 'bert-base-uncased']:
            for balance in ['', '_balance']:
                df.loc[i] = [dataset_name, model_name, tag, 0, 0, 0, 0, 0]
                logger.info(f'{dataset_name}, {model_name}, {tag}, {balance}')

                filename = f'embed_{model_name.replace("/", "_")}_{"CLS" if support_CLS else "Avg"}'
                train_data_path = f'./benchmark/{dataset_name}/{filename}.pkl'
                model_path = f"pretrained/{dataset_name}/{filename.replace('/', '_')}_{tag}{balance}.pth"
                directory = f'pretrained/{dataset_name}'
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # read embedding features
                with open(train_data_path, "rb") as input_file:
                    X = pickle.load(input_file)['feature'].to_numpy()

                # read original dataset
                excel_file_path = f"rawdata/{dataset_name}.csv"
                data = pd.read_csv(excel_file_path)

                # each raw of X is a vector
                X = np.vstack(X)
                y = data['normal'].to_numpy()

                if tag:
                    indice = (data[tag] == 1) | (data['normal'] == 1)
                    X, y = X[indice, :], y[indice]

                # Balance the data by randomly sampling from the larger set
                if balance:
                    min_data_size = min(sum(y == 0), sum(y == 1))
                    indices_hate = np.nonzero(y == 0)[0]
                    np.random.shuffle(indices_hate)
                    indices_normal = np.nonzero(y == 1)[0]
                    np.random.shuffle(indices_normal)
                    indices = np.concatenate([indices_hate[:min_data_size], indices_normal[:min_data_size]])
                    X, y = X[indices, :], y[indices]

                full_dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

                n_tot = len(full_dataset)
                train_size = int(0.6 * n_tot)
                val_size = int(0.2 * n_tot)
                test_size = n_tot - train_size - val_size

                ind = shuffle(list(range(n_tot)))
                train_ind = ind[:train_size]
                val_ind = ind[train_size:train_size + val_size]
                test_ind = ind[train_size + val_size:]

                train_dataset = TensorDataset(torch.tensor(X[train_ind]), torch.tensor(y[train_ind]))
                # val_dataset = TensorDataset(torch.tensor(X[val_ind]), torch.tensor(y[val_ind]))
                generator1 = torch.Generator().manual_seed(42)
                generator2 = torch.Generator().manual_seed(12)
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.7, 0.3],
                                                                           generator=generator2)
                test_dataset = TensorDataset(torch.tensor(X[test_ind]), torch.tensor(y[test_ind]))

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                df.loc[i, 'training size'] = train_size
                df.loc[i, 'class ratio'] = 1 if balance else sum(y[train_ind] == 1)/sum(y[train_ind] == 0)

                classifier = BinaryClassifier(X.shape[1], 256).to(device)
                num_epochs = 20
                criterion = nn.BCELoss()

                classifier = train(classifier, train_loader, val_loader, verbose=True, criterion=criterion,
                                   device=device, num_epochs=num_epochs, model_path=model_path)

                test_crit = ZeroOneLoss
                df.loc[i, 'accuracy'] = 1-test(classifier, test_loader, device=device, criterion=test_crit)
                pred = classifier(torch.Tensor(X[test_ind]).to(device)).detach().cpu().squeeze().numpy()
                df.loc[i, 'f1'] = f1_score(y[test_ind], pred>0.5)
                df.loc[i, 'auc'] = roc_auc_score(y[test_ind], pred)
                i += 1

    df.to_csv('./benchmark/pretrain_model.csv', index=False)


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
