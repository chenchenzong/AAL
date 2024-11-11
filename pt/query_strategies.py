import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader



def random_query(data_loader, query_size=10):
    sample_idx = []

    # Because the data has already been shuffled inside the data loader,
    # we can simply return the `query_size` first samples from it
    for batch in data_loader:

        _, _, idx = batch
        sample_idx.extend(idx.tolist())

        if len(sample_idx) >= query_size:
            break

    return sample_idx[0:query_size]


def least_confidence_query(model, device, data_loader, query_size=10):
    confidences = []
    indices = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)

            # Keep only the top class confidence for each sample
            most_probable = torch.max(probabilities, dim=1)[0]
            confidences.extend(most_probable.cpu().tolist())
            indices.extend(idx.tolist())

    conf = np.asarray(confidences)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(conf)
    # Return the indices corresponding to the lowest `query_size` confidences
    return ind[sorted_pool][0:query_size]

def entropy_query(model, device, data_loader, query_size=10):
    entropy_list = []
    indices = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            data, _, idx = batch

            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)

            # Keep only the top class confidence for each sample
            most_probable = torch.max(probabilities, dim=1)[0].cpu()
            entropy = most_probable*np.log(most_probable)
            entropy_list.extend(entropy.tolist())
            indices.extend(idx.tolist())

    conf = np.asarray(entropy_list)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(conf)
    # Return the indices corresponding to the lowest `query_size` confidences
    return ind[sorted_pool][0:query_size]

def margin_query(model, device, data_loader, query_size=10):
    margin_list = []
    indices = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            data, _, idx = batch
            logits = model(data.to(device))
            probabilities = F.softmax(logits, dim=1)
            spv = np.shape(probabilities)
            # Keep only the top class confidence for each sample
            pat = np.partition(probabilities.cpu(), (spv[1] - 2, spv[1] - 1), axis=1)
            margin = pat[:, spv[1] - 1] - pat[:, spv[1] - 2]
            margin_list.extend(margin.tolist())
            indices.extend(idx.tolist())

    conf = np.asarray(margin_list)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(conf)
    # Return the indices corresponding to the lowest `query_size` confidences
    return ind[sorted_pool][0:query_size]
