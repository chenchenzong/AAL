import pickle
import os
import sys
import yaml
from easydict import EasyDict

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
from Pool_Manager import Pool_Manager
from Manager import Manager
from Server import Server
from Worker import Worker
import argparse

def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('method', type=str)
    p.add_argument('assignment_type', type=str, default='DA')
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
    #torch.save(model.state_dict(), config.init_model_path)

    # load the indices:
    if config.initial_idx_path is not None:
        with open(config.initial_idx_path, 'rb') as f:
            labeled_idx = pickle.load(f)
    else:
        print("No Initial Indices Found - Drawing Random Indices...")
        unlabeled_idx = np.nonzero(train_set.unlabeled_mask)[0]
        labeled_idx = np.random.choice(unlabeled_idx, config.initial_size, replace=False)
        # save the results:
        #with open('idx.pkl', 'wb') as f:
        #    pickle.dump(labeled_idx, f)
    train_set.update_label(labeled_idx)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        config.lr,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True)

    model_path = config.first_train_model_path
    #model.load_state_dict(torch.load(config.first_train_model_path))

    unlabeled_idx = np.nonzero(train_set.unlabeled_mask)[0]
    split_list = []

    ###
    unlabel_set_num = config.worker_num

    if config.split_strategy == 'RS':
        np.random.shuffle(unlabeled_idx)
        step = unlabeled_idx.shape[0] // unlabel_set_num
        for i in range(unlabel_set_num):
            if i == unlabel_set_num - 1:
                split_list.append(unlabeled_idx[i*step:])
            else:
                split_list.append(unlabeled_idx[i*step:(i+1)*step])
    elif config.split_strategy == 'DS':
        pass

    ######################### 线程定义 ####################################
    ## 1
    pool_m = Pool_Manager("LabelPool_Manager", train_set, test_set)

    ## 2
    server_dict = {}
    for i in range(config.server_num):
        server_name = 'server-' + str(i+1)
        server_dict[server_name] = Server(server_name, pool_m)

    ## 3
    method_list = ['margin', 'least_confidence', 'entropy', 'random']
    worker_dict = {}
    for i in range(config.worker_num):
        worker_name = 'worker-' + str(i+1)
        #Definite Assignment
        if args.assignment_type == 'DA':
            worker_dict[worker_name] = Worker(worker_name, pool_m, split_list[i], model_path, config.worker_query_num,
                                              config.query_size, args.method, device)
        #Random Assignment
        elif args.assignment_type == 'RA':
            #method = method_list[np.random.randint(0, 3)]
            method = method_list[i % 3]
            worker_dict[worker_name] = Worker(worker_name, pool_m, split_list[i], model_path, config.worker_query_num,
                                              config.query_size, method, device)

    ## 4
    manager = Manager("Manager", pool_m, server_dict, worker_dict, config.worker_query_num, config.query_size)

    ######################### 开启线程 ####################################
    pool_m.start()
    for i in range(config.server_num):
        server_name = 'server-' + str(i+1)
        server_dict[server_name].start()
    for i in range(config.worker_num):
        worker_name = 'worker-' + str(i+1)
        worker_dict[worker_name].start()
    manager.start()
    for i in range(config.server_num):
        server_name = 'server-' + str(i+1)
        server_dict[server_name].join()
    for i in range(config.worker_num):
        worker_name = 'worker-' + str(i+1)
        worker_dict[worker_name].join()
    manager.join()
    pool_m.join()
