import pickle
import os
import sys
import argparse
from keras.utils import to_categorical
from sklearn.datasets import load_boston, load_diabetes
from models import *
from query_methods import *
import keras
from DataLoad import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.models import load_model
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
config = tf.ConfigProto()
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('ID', type=str)
    p.add_argument('QueryStrategy', type=str)
    args = p.parse_args()
    return args

######################### config ####################################

args_data_type = 'cifar10'

######################### 数据集加载及预处理 ####################################

if args_data_type == 'cifar10':
    (X_train, Y_train), (X_test, Y_test) = load_cifar_10()
    num_labels = 10
    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, 3)
    else:
        input_shape = (3, 32, 32)
    evaluation_function = train_cifar10_model
if args_data_type == 'cifar100':
    (X_train, Y_train), (X_test, Y_test) = load_cifar_100()
    num_labels = 100
    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, 3)
    else:
        input_shape = (3, 32, 32)
    evaluation_function = train_cifar100_model

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
XandY_train = np.hstack((X_train, Y_train))
XandY_test = np.hstack((X_test, Y_test))

with open("val_idx-5000.pkl", 'rb') as f:
        val_idx = pickle.load(f)
index = np.arange(X_train.shape[0])
train_idx = index[np.logical_not(np.isin(index, val_idx))]
XandY_val = XandY_train[val_idx]
XandY_train = XandY_train[train_idx]
X_val = XandY_val[:,:-1]
Y_val = XandY_val[:,-1]
X_train = XandY_train[:, :-1]
Y_train = XandY_train[:, -1]

######################### 模型加载并查询 ####################################
args = parse_input()
model = load_model('worker.h5')

with open(args.ID + "_lab_ind.pkl", 'rb') as f:
    labeled_idx = pickle.load(f)


if args.QueryStrategy == 'Random':
    method = RandomSampling
elif args.QueryStrategy == 'Uncertainty':
    method = UncertaintySampling
elif args.QueryStrategy == 'UncertaintyEntropy':
    method = UncertaintyEntropySampling
elif args.QueryStrategy == "UncertaintyMargin":
    method = UncertaintyMarginSampling


query_method = method(None, (32, 32, 3), 10, 1)
query_method.update_model(model)

old_labeled = np.copy(labeled_idx)
labeled_idx = query_method.query(X_train.reshape(-1,32,32,3), to_categorical(Y_train), labeled_idx, 500)

new_idx = labeled_idx[np.logical_not(np.isin(labeled_idx, old_labeled))]

if os.path.exists(args.ID + "_new_ind.pkl") == True:
    os.remove(args.ID + "_new_ind.pkl")

with open(args.ID + "_new_ind.pkl", 'wb') as f:
    pickle.dump(new_idx, f)