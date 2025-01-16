import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from aug_utils import *
from data_utils import *
from data_utils_v2 import *
from training_utils import EarlyStopping, SMAPE


def get_current_script_name():

    return os.path.basename(__file__)


def get_aug_bool():
    scriptn = get_current_script_name()
    if "_Aug.py" in scriptn:
        return True
    else:
        return False


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description=" training script")
    parser.add_argument("--MODEL", type=str, default="CNNLSTM")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=512,
                        help="Number of hidden_dim for the model")
    parser.add_argument("--past_steps",
                        type=int,
                        default=10,
                        help="Input window ")
    parser.add_argument("--future_steps", type=int, default=1, help=" window ")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=1500,
                        help="Number of training epochs")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.6,
                        help="Dropout rate")
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--SEED", type=int, default=0)
    parser.add_argument("--patience", default=3)
    parser.add_argument("--time_col", default="date")
    parser.add_argument("--target_col", default="total_index")
    parser.add_argument("--augment", type=str2bool, default=get_aug_bool())

    args = parser.parse_args()
    config = vars(args)
    fpath = "../data/fred/Combined_CPI_Korea_After_2010.csv"
    df = pd.read_csv(fpath)
    print("df", df)
    df = multi_aug(config, args, df, method="linear")
    print("df", df)
    df.to_csv(fpath.replace("/fred/", "/fred_aug/"))
