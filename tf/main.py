from models import *
from query_methods import *
import pickle
import os
import keras
from keras.utils import to_categorical
from DataLoad import *
from Server import Server
from Worker import Worker
from Manager import Manager_Worker
from Pool_Manage import Pool_Manager
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import tensorflow as tf
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(session)


def main(AssignType, QueryStrategy = None):
    ######################### config ####################################
    WorkerQueryNum = 5					# 单个worker查询次数
    ServerTrainNum = 5					# server训练次数
    ServerTrainThreshold = 1000         # worker查询增量为多少时server才更新
    Assignment_Type = AssignType        # worker查询策略分配方式：RA随机分配查询策略，DA给定查询策略（所有worker策略一致）
    Query_Strategy = QueryStrategy      # DA下给定的查询策略
    server_num = 2                      
    worker_num = 2                     
    Split_Strategy = 'RS'               # 将数据集划分为各个worker的方式：RS随机进行划分, DS通过聚类进行划分（实验中DS效果并不好）
    args_data_type = "cifar10"          
    args_batch_size = 500               #一个worker单次查询数量
    args_initial_size = 5000
    num_labels = 10
    args_val_size = 5000

    #无需修改
    unlabel_set_num = worker_num
    serverId_head = 'Server'
    WorkerID_head = "Worker"
    modelVersion = 0
    args_experiment_folder = "result"



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

    #val_idx = np.random.choice(X_train.shape[0], args_val_size, replace=False)
    #with open("val_idx-5000.pkl", 'wb') as f:
    #    pickle.dump(val_idx, f)
    with open("val_idx-5000.pkl", 'rb') as f:
        val_idx = pickle.load(f)
    index = np.arange(X_train.shape[0])
    train_idx = index[np.logical_not(np.isin(index, val_idx))]
    XandY_val = XandY_train[val_idx]
    XandY_train = XandY_train[train_idx]
    X_val = XandY_val[:,:-1]
    Y_val = XandY_val[:,-1]
    X_train = XandY_train[:,:-1]
    Y_train = XandY_train[:,-1]

    # load the indices:
    #labeled_idx = np.random.choice(X_train.shape[0], args_initial_size, replace=False)
    #np.random.shuffle(labeled_idx)
    #with open("labeled_idx-5000.pkl", 'wb') as f:
    #    pickle.dump(labeled_idx, f)
    with open("labeled_idx-5000.pkl", 'rb') as f:
        labeled_idx = pickle.load(f)

    index_list = np.arange(X_train.shape[0])
    unlabeled_idx = index_list[np.logical_not(np.isin(index_list, labeled_idx))]

    ######################### create the checkpoint path ####################################
    if not os.path.isdir(os.path.join(args_experiment_folder, 'models')):
        os.mkdir(os.path.join(args_experiment_folder, 'models'))
    model_folder = os.path.join(args_experiment_folder, 'models')

    checkpoint_path = os.path.join(model_folder, '{alg}_{datatype}_{init}_{batch_size}.hdf5'.format(
        alg=Query_Strategy, datatype=args_data_type, batch_size=args_batch_size, init=args_initial_size
    ))

    ######################### 定义模型并预训练 ####################################
    #init_acc, model = evaluate_sample(evaluation_function, X_train[labeled_idx, :].reshape(-1,32,32,3), to_categorical(Y_train[labeled_idx]), X_test.reshape(-1,32,32,3), to_categorical(Y_test),X_val.reshape(-1,32,32,3), to_categorical(Y_val),
    #                             checkpoint_path)
    #model.save('base_model.h5')
    model = load_model("base_model.h5")
    model.save('worker.h5', overwrite=True)
    loss, init_acc = model.evaluate(X_test.reshape(-1,32,32,3), to_categorical(Y_test), verbose=0)
    print("Test Accuracy Is " + str(init_acc))



    ######################### Unlabel Set Split ####################################
    split_list = []
    # Random Split
    if Split_Strategy == "RS":
        np.random.shuffle(unlabeled_idx)
        step = unlabeled_idx.shape[0] // unlabel_set_num
        for i in range(unlabel_set_num):
            if i == unlabel_set_num - 1:
                split_list.append(unlabeled_idx[i * step:])
            else:
                split_list.append(unlabeled_idx[i * step:(i + 1) * step])
    # Diverse Split
    else:
        kmeans = KMeans(n_clusters=unlabel_set_num)
        result = kmeans.fit(X_train[unlabeled_idx,:]).labels_.tolist()
        for i in range(unlabel_set_num):
            split_index = [t for t, v in enumerate(result) if v == i]
            split_list.append(unlabeled_idx[split_index])   ####存疑
            print(len(split_index))



    ######################### 线程定义 ####################################
    ## 1
    Pool_M = Pool_Manager("LabelPool_Manager", XandY_train, XandY_test, labeled_idx, unlabeled_idx)

    ## 2
    server_dict = {}
    for i in range(server_num):
        server_name = serverId_head + str(i + 1)
        server_dict[server_name] = Server(server_name, Pool_M, model,
                                          modelVersion, Assignment_Type, Query_Strategy,
                                          Split_Strategy, init_acc, evaluation_function, checkpoint_path, session)
    ## 3
    QueryStrategy_List = ["UncertaintyEntropy", 'Uncertainty', "UncertaintyMargin"]

    worker_dict = {}
    for i in range(worker_num):
        worker_name = WorkerID_head + str(i + 1)
        # Definite Assignment
        if Assignment_Type == "DA":
            base_model_index = np.random.randint(0, server_num)
            server_name = serverId_head + str(base_model_index + 1)
            worker_dict[worker_name] = Worker(worker_name, Pool_M, split_list[i], model, modelVersion, server_dict, serverId_head, server_name, WorkerQueryNum, Query_Strategy, session, num_labels)
        # Random Assignment
        else:
            strategy = QueryStrategy_List[np.random.randint(0, 4)]
            base_model_index = np.random.randint(0, server_num)
            server_name = serverId_head + str(base_model_index + 1)
            worker_dict[worker_name] = Worker(worker_name, Pool_M, split_list[i], model,
                                              modelVersion,
                                              server_dict, serverId_head, server_name,
                                              WorkerQueryNum, strategy, session, num_labels)
    ## 4
    Manager = Manager_Worker("Manager", server_dict, serverId_head, worker_dict, WorkerID_head, ServerTrainNum, ServerTrainThreshold)

    ######################### 开启线程 ####################################
    Pool_M.start()
    for i in range(server_num):
        server_name = serverId_head + str(i + 1)
        server_dict[server_name].start()
    for i in range(worker_num):
        worker_name = WorkerID_head + str(i + 1)
        worker_dict[worker_name].start()
    Manager.start()
    for i in range(server_num):
        server_name = serverId_head + str(i + 1)
        server_dict[server_name].join()
    for i in range(worker_num):
        worker_name = WorkerID_head + str(i + 1)
        worker_dict[worker_name].join()
    Manager.join()
    Pool_M.join()

if __name__ == '__main__':
    main("DA", "UncertaintyMargin")
    #main("DA", "UncertaintyEntropy")
    #main("DA", 'Uncertainty')
    #main("RA")







