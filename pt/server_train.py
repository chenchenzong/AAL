import pickle
import os
import sys
import yaml
from easydict import EasyDict
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from models import *
from utils import *
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('idx_file', type=str)
    p.add_argument('model_path', type=str)
    p.add_argument('acc_path', type=str)
    args = p.parse_args()
    return args

if __name__ == '__main__':

    args = parse_input()

    # read config from yaml file
    with open('A_config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)

    # load the dataset:
    if config.data_type == 'cifar10':
        transform_train = transforms.Compose(data_augmentation(config.data_type))
        transform_test = transforms.Compose(data_augmentation(config.data_type))
        train_set = IndexedDataset(config.data_type, transform_train)
        test_set = IndexedDataset(config.data_type, transform_test, test=True)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

        num_labels = 10
    elif config.data_type == 'cifar100':
        transform_train = transforms.Compose(data_augmentation(config.data_type))
        transform_test = transforms.Compose(data_augmentation(config.data_type))
        train_set = IndexedDataset(config.data_type, transform_train)
        test_set = IndexedDataset(config.data_type, transform_test, test=True)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

        num_labels = 100

    model = vgg16(num_labels)

    # CPU or GPU
    device = 'cuda' if config.gpu else 'cpu'
    model.to(device)

    # load the indices:
    with open(args.idx_file, 'rb') as f:
        labeled_idx = pickle.load(f)

    train_set.update_label(labeled_idx)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        config.lr,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True)

    acc, model_path = one_training_process(args.model_path, criterion, optimizer, train_set, test_set, config.batch_size, model, device)
    model.load_state_dict(torch.load(model_path))
    np.savetxt(args.acc_path, [acc], fmt="%.4f")
