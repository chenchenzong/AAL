from models import *
from query_methods import *
import pickle
import os
import keras
from keras.utils import to_categorical
from DataLoad import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(session)

######################### config ####################################

args_data_type = 'cifar10'
args_experiment_folder = "server_result"

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

######################### create the checkpoint path ####################################

if not os.path.isdir(os.path.join(args_experiment_folder, 'models')):
    os.mkdir(os.path.join(args_experiment_folder, 'models'))
model_folder = os.path.join(args_experiment_folder, 'models')

checkpoint_path = os.path.join(model_folder, '{datatype}.hdf5'.format(datatype=args_data_type ))

######################### 模型训练并保存 ####################################

# load the indices:
with open("server_train_lab_ind.pkl", 'rb') as f:
    labeled_idx = pickle.load(f)

with session.as_default():
    with session.graph.as_default():
        acc, model = evaluate_sample(evaluation_function, X_train[labeled_idx, :].reshape(-1,32,32,3), to_categorical(Y_train[labeled_idx]), X_test.reshape(-1,32,32,3), to_categorical(Y_test), X_val.reshape(-1,32,32,3), to_categorical(Y_val),
                                 checkpoint_path)

        model.save_weights('server.h5', overwrite=True)
        model.save('worker.h5', overwrite=True)
