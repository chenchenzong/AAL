import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from termcolor import colored
from models import *
from query_methods import *
from DataLoad import *
from keras import optimizers
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from keras.utils import to_categorical
import keras.backend as K
from keras.models import Model
import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2


class Server(threading.Thread):
    def __init__(self, ID, Pool_M, model, modelVersion, Assignment_Type, Query_Strategy, Split_Strategy, init_acc, evaluation_function, checkpoint_path, session):
        threading.Thread.__init__(self)
        self.ID = ID
        self.Pool_M = Pool_M
        self.model = model
        self.modelVersion = modelVersion
        self.Query_Strategy = Assignment_Type
        if Query_Strategy == None:
            self.QS = " "
        else:
            self.QS = Query_Strategy
        self.Split_Strategy = Split_Strategy
        self.score = init_acc      
        self.evaluation_function = evaluation_function
        self.checkpoint_path = checkpoint_path

        self.score_list = []
        self.query_num = 0
        self.update = False              # server是否正在更新
        self.stop = False                # server是否继续监听
        self.modelUpdateSignal = False   #server是否更新结束
        
        self.score_list.append([self.query_num, self.score])

        #### 这个地方需要运行一次模型，不运行会报错，有点没搞明白，但下面这个部分对实际模型的训练没有影响
        self.session = tf.Session(config=config)
        keras.backend.tensorflow_backend.set_session(self.session)
        self.model = get_VGG_model(input_shape=(32, 32, 3), labels=10)
        optimizer = optimizers.Adam()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.callbacks = [DelayedModelCheckpoint(filepath=self.checkpoint_path, verbose=1, weights=True)]
        self.TrainPool = self.Pool_M.TrainPool[self.Pool_M.label_ind, :]
        X_train = self.TrainPool[:, :-1].reshape(-1, 32, 32, 3)
        Y_train = to_categorical(self.TrainPool[:, -1:])
        self.model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=0)

        self.f = K.function([self.model.layers[0].input, K.learning_phase()],
                            [self.model.layers[-1].output])



    # overrides run
    def run(self):
        print(colored('开始线程: ' + self.ID, "red"))
        self.listen()
        print(colored('结束线程: ' + self.ID, "red"))

    # 监听函数
    def listen(self):
        while self.stop == False:
            if self.update == True:
                with self.session.as_default():
                    with self.session.graph.as_default():
                        print(colored('start train-------, modelVesion: ' + str(self.ID) + " ,modelVersion: " + str(
                            self.modelVersion) + ',poolShape: ' + str(self.Pool_M.label_ind.shape[0]), "red"))
                        self.TrainPool = self.Pool_M.TrainPool[self.Pool_M.label_ind, :]
                        while self.update == True:
                            self.train()
                        print(colored(
                            'end train---------' + str(self.ID) + " ,modelVersion: " + str(
                                self.modelVersion) + ',score: ' + str(
                                self.score), "red"))


    def train(self):
        # 一旦检测到标记池有更新，则需开始训练模型
        try:
            ######################### 索引保存并驱动server进行训练 ####################################

            if os.path.exists('server.h5') == True:
                os.remove('server.h5')
            with open("server_train_lab_ind.pkl", 'wb') as f:
                pickle.dump(self.Pool_M.label_ind, f)
            print("start training........")
            os.system("python server_train.py")
            print("......")

            ######################### 等待训练结果 ####################################
            while os.path.exists('server.h5') != True:
                print("waiting........")
                time.sleep(10)
            self.model.load_weights('server.h5')
            time.sleep(20)
            loss, self.score = self.model.evaluate(self.Pool_M.TestPool[:, :-1].reshape(-1, 32, 32, 3),
                                               to_categorical(self.Pool_M.TestPool[:, -1:]), verbose=0)

            self.query_num += 1
            self.score_list.append([self.query_num, self.score])
            self.modelUpdateSignal = True
            self.modelVersion += 1
            self.update = False
        except Exception as e:
            print(colored('ERROR!', "red"))
            pass

    # worker一旦检测到模型更新好了，立马请求最新模型
    def returnModel(self):
        self.modelUpdateSignal = False
        return self.modelVersion, self.ID
