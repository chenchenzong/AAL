import numpy as np
import time
import threading
import math
import Server
import random
from termcolor import colored
from utils import *
import pickle
import os


class Worker(threading.Thread):
    def __init__(self, ID, Pool_M, unlab_ind, model_path, AllQueryNum, query_size, method, device, modelVersion=0):
        threading.Thread.__init__(self)
        self.ID = ID
        self.server_name = 'init'
        self.Pool_M = Pool_M
        self.model_path = model_path
        self.unlab_ind = unlab_ind
        self.modelVersion = modelVersion
        self.AllQueryNum = AllQueryNum   # 总共查询的次数
        self.method = method
        self.device = device
        self.query_size = query_size
        self.queryNum = 0
        self.worker_file = self.Pool_M.filename + '/worker/'

    # overrides run
    def run(self):
        print(colored('开始线程: ' + self.ID, "blue"))
        self.listen()
        print(colored('结束线程: ' + self.ID, "blue"))

    # 监听函数
    def listen(self):
        while self.queryNum < self.AllQueryNum:
            time.sleep(random.randint(0, 60))
            self.query()
            self.updateModel()

    def query(self):
        unidx_file = self.worker_file + self.ID + '_' + str(self.queryNum) + "_lab_ind.pkl"
        sample_idx_file = self.worker_file + self.ID + '_' + str(self.queryNum) + "_sample_ind.pkl"
        with open(unidx_file, 'wb') as f:
            pickle.dump(self.unlab_ind, f)

        print("start querying........")
        os.system("python worker_query.py " + unidx_file + " " + self.method + " " + sample_idx_file + " " + self.model_path)
        print("......")

        while not os.path.exists(sample_idx_file):
            print("worker waiting........")
            time.sleep(10)

        with open(sample_idx_file, 'rb') as f:
            sample_idx = pickle.load(f)

        self.unlab_ind = self.unlab_ind[np.logical_not(np.isin(self.unlab_ind, sample_idx))]
        self.queryNum += 1
        print(colored(
                'Thread-' + self.ID + ',QueryNum: ' + str(
                    self.queryNum) + ',modelVesion：' + self.server_name + "-" + str(
                    self.modelVersion) + ', unlabel-poolShape: ' + str(len(self.unlab_ind)), "blue"))
        self.Pool_M.submitLabelData(sample_idx)

    def updateModel(self):
        while self.modelVersion >= self.Pool_M.latest_model_version:
            time.sleep(1)
        if self.Pool_M.latest_model_path != '':
            self.modelVersion = self.Pool_M.latest_model_version
            self.model_path = self.Pool_M.latest_model_path
            self.server_name = self.Pool_M.latest_mv_server
        else:
            print('load latest model error!')

