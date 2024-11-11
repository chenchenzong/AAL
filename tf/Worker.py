import numpy as np
import time
import threading
import math
import Server
import random
from termcolor import colored
from models import *
from query_methods import *
from DataLoad import *
from keras.utils import to_categorical
import keras.backend as K
from keras.models import Model
import warnings
warnings.filterwarnings("ignore")

class Worker(threading.Thread):
    def __init__(self, ID, Pool_M, unlab_ind, model, modelVersion, server_dict, severId_head, server_name, AllQueryNum, Query_Strategy, session, num_labels):
        threading.Thread.__init__(self)
        self.ID = ID
        self.Pool_M = Pool_M
        self.unlab_ind = unlab_ind
        self.model = model
        self.modelVersion = modelVersion
        self.server_dict = server_dict
        self.serverId_head = severId_head
        self.server_name = server_name
        self.AllQueryNum = AllQueryNum   # 总共查询的次数
        self.Query_Strategy = Query_Strategy

        self.stop = False  # worker是否继续监听
        self.wait = False  # worker是否处于等待状态
        self.fail = False  # 查询是否失败
        self.first_train = True
        self.num_labels = num_labels
        self.queryNum = 0

        self.session = session
        self.K = K


    # overrides run
    def run(self):
        print(colored('开始线程: ' + self.ID, "blue"))
        self.listen()
        print(colored('结束线程: ' + self.ID, "blue"))

    # 监听函数
    ###
    ### 当前设置为worker查询一次即等待模型更新，如若需要worker查询多次，这里需要修改一下
    ###
    def listen(self):
        while self.stop == False:
            self.updateModel()
            while self.queryNum <= self.AllQueryNum and self.wait == False:
                self.query()

                #由于查询速度相对较快，初始时由于直接加载预训练模型可直接查询，之后通过wait参数等待模型更新
                if self.fail == False and self.first_train == True:
                    #self.wait = True
                    self.first_train = False
                elif self.fail == False and self.first_train == False:
                    self.wait = True

    '''
    ### worker查询多次的样例，假如一次模型更新期间可以查询20次
    def listen(self):
        while self.stop == False:
            self.updateModel()
            i = 1
            while self.queryNum <= self.AllQueryNum and self.wait == False:
                self.query()
				
				## 由于初始加载模型查询20次后才开始训练，所以其实是查询40次才能等到第一次模型更新结束，之后都是20次即可
                if self.fail == False and i < 20:
                    i += 1
                elif self.fail == False and self.first_train == True and i < 40:
                    if i == 20:   			
                        time.sleep(60)  
                    i += 1
                    if i == 40:
                        self.first_train = False
                elif self.fail == False and self.first_train == False:
                    self.wait = True

	'''

    def query(self):
        try:

            ######################### 索引保存并驱动worker进行查询 ####################################

            labeled_idx = np.arange(self.Pool_M.TrainPool.shape[0])[
                    np.logical_not(np.in1d(np.arange(self.Pool_M.TrainPool.shape[0]), self.unlab_ind))]

            if os.path.exists(self.ID + "_lab_ind.pkl") == True:
                os.remove(self.ID + "_lab_ind.pkl")
            with open(self.ID + "_lab_ind.pkl", 'wb') as f:
                pickle.dump(labeled_idx, f)

            print("start query........")
            os.system("python worker_query.py " + self.ID + " " + self.Query_Strategy)
            print("......")

            ######################### 等待查询结果 ####################################
            while os.path.exists(self.ID + "_new_ind.pkl") != True:
                print("waiting........")
                time.sleep(10)

            with open(self.ID + "_new_ind.pkl", 'rb') as f:
                new_idx = pickle.load(f)

            self.Pool_M.submitLabelData(new_idx)
            # 将该数据从UnlabelPool中删除
            self.unlab_ind = self.unlab_ind[np.logical_not(np.isin(self.unlab_ind, new_idx))]

            ######################### 查询情况输出 ####################################
            self.queryNum += 1
            print(colored(
                    'Thread-' + self.ID + ',QueryNum: ' + str(
                        self.queryNum) + ',modelVesion：' + self.server_name + "-" + str(
                        self.modelVersion) + ', unlabel-poolShape: ' + str(len(self.unlab_ind)), "blue"))
            self.fail = False
        except Exception as e:
            self.fail = True

    # 更新模型, worker一旦检测到模型更新好了，立马请求最新模型
    # 当modelversion=0时，最新的模型为server编号最大且modelversion为1的
    # 当modelversion！=0时，最新的模型modelversion可能与现在modelversion相同，也可能比现在大1。 由此分为以下三种情况
    
    def updateModel(self):
        index = int(self.server_name.replace("Server", ""))
        if self.modelVersion == 0:
            for i in range(len(self.server_dict) - 1, -1, -1):
                server_name = self.serverId_head + str(i + 1)
                if self.server_dict[server_name].modelVersion == self.modelVersion + 1:
                    self.modelVersion, self.server_name = self.server_dict[server_name].returnModel()

                    self.wait = False
                    break
        else:
            server_name = self.serverId_head + str(1)
            if self.server_dict[server_name].modelVersion != self.modelVersion + 1:
                for i in range(len(self.server_dict) - 1, index - 1, -1):
                    server_name = self.serverId_head + str(i + 1)
                    if self.server_dict[server_name].modelVersion == self.modelVersion:
                        self.modelVersion, self.server_name = self.server_dict[server_name].returnModel()

                        self.wait = False
                        break
            else:
                for i in range(len(self.server_dict) - 1, -1, -1):
                    server_name = self.serverId_head + str(i + 1)
                    if self.server_dict[server_name].modelVersion == self.modelVersion + 1:
                        self.modelVersion, self.server_name = self.server_dict[server_name].returnModel()

                        self.wait = False
                        break

