import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import time
from pytorchtools import EarlyStopping
from torch.utils.data import SubsetRandomSampler
from query_strategies import *

def data_augmentation(data_type):
    aug = []
    aug.append(transforms.ToTensor())
    # normalize  [- mean / std]
    if data_type == 'cifar10':
        aug.append(transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    elif data_type == 'cifar100':
        aug.append(transforms.Normalize(
                #(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))

    return aug

class IndexedDataset(Dataset):

    def __init__(self, data_type, transform=None, test=False):
        self.data_type = data_type
        self.transform = transform
        if self.data_type == 'cifar10':
            if test:
                testset = torchvision.datasets.CIFAR10(
                    root='./data/cifar10', train=False,
                    download=True, transform=transform
                )
                self.data = testset.data
                self.labels = testset.targets
                self.unlabeled_mask = np.zeros(len(self.data))
            else:
                trainset = torchvision.datasets.CIFAR10(
                    root='./data/cifar10', train=True,
                    download=True, transform=transform
                )
                self.data = trainset.data
                self.labels = trainset.targets
                self.unlabeled_mask = np.ones(len(self.data))
        elif self.data_type == 'cifar100':
            if test:
                testset = torchvision.datasets.CIFAR100(
                    root='./data/cifar100', train=False,
                    download=True, transform=transform
                )
                self.data = testset.data
                self.labels = testset.targets
                self.unlabeled_mask = np.zeros(len(self.data))
            else:
                trainset = torchvision.datasets.CIFAR100(
                    root='./data/cifar100', train=True,
                    download=True, transform=transform
                )
                self.data = trainset.data
                self.labels = trainset.targets
                self.unlabeled_mask = np.ones(len(self.data))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data, self.labels[idx], idx

    def update_label(self, idx_list):
        self.unlabeled_mask[idx_list] = 0
        return

def train(train_loader, model, criterion, optimizer, epoch, epochs, device):
    start = time.time()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print(" === Epoch: [{}/{}] === ".format(epoch + 1, epochs))
    for batch_index, (inputs, targets, _) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))
    end = time.time()
    print("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total
    return train_loss, train_acc

def test(test_loader, model, criterion, epoch, epochs, best_prec, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    print(" === Validate ===".format(epoch + 1, epochs))
    with torch.no_grad():
        for batch_index, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    acc = 100. * correct / total
    is_best = acc > best_prec
    if is_best:
        best_prec = acc
    return test_acc, best_prec

def one_training_process(checkpoint_path, criterion, optimizer, train_set, test_set, batch_size, model, device, query_step=-1):
    epochs = 200
    early_stopping = EarlyStopping(patience=200, verbose=True)
    last_epoch = -1
    best_prec = 0
    #base_lr = get_current_lr(optimizer)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    labeled_idx = [i for i, x in enumerate(train_set.unlabeled_mask) if x == 0]
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=SubsetRandomSampler(labeled_idx))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print("            =======  Training  =======\n")
    vaild_acc, best_prec = test(test_loader, model, criterion, -1, epochs, best_prec, device)
    for epoch in range(last_epoch + 1, epochs):
        train(train_loader, model, criterion, optimizer, epoch, epochs, device)
        vaild_acc, best_prec = test(test_loader, model, criterion, epoch, epochs, best_prec, device)
        scheduler.step()
        #adjust_learning_rate(optimizer, epoch, base_lr)
        early_stopping(vaild_acc, model,
                       path=checkpoint_path + str(query_step) + '.pth.tar')
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model_path = checkpoint_path + str(query_step) + '.pth.tar'
    return best_prec, model_path

def query_the_oracle(dataset, model, query_strategy, device, query_size, batch_size=256):
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]
    print("Unlabel pool size is " + str(len(unlabeled_idx)))
    pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=SubsetRandomSampler(unlabeled_idx))
    if query_strategy == 'random':
        sample_idx = random_query(pool_loader, query_size)
    elif query_strategy == 'margin':
        sample_idx = margin_query(model, device, pool_loader, query_size)
    elif query_strategy == 'least_confidence':
        sample_idx = least_confidence_query(model, device, pool_loader, query_size)
    elif query_strategy == 'entropy':
        sample_idx = entropy_query(model, device, pool_loader, query_size)
    dataset.update_label(sample_idx)
    return sample_idx

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def reset_current_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, base_lr):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = base_lr
    elif epoch < 120:
        lr = base_lr * 0.1
    else:
        lr = base_lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr