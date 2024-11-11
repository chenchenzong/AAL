import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from termcolor import colored
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import pickle
import os


class Server(threading.Thread):
    def __init__(self, ID, Pool_M):
        threading.Thread.__init__(self)
        self.ID = ID
        self.Pool_M = Pool_M
        self.modelVersion = 0

        self.update = False              # server是否正在更新
        self.running = True                # server是否继续监听
        self.server_file = self.Pool_M.filename + '/server/'

    # overrides run
    def run(self):
        print(colored('开始线程: ' + self.ID, "red"))
        self.listen()
        print(colored('结束线程: ' + self.ID, "red"))

    # 监听函数
    def listen(self):
        while self.running:
            if self.update:
                print(colored('start train-------, modelVesion: ' + str(self.ID) + " ,modelVersion: " + str(
                    self.modelVersion) + ',poolShape: ' + str(len(self.Pool_M.label_ind)), "red"))

                idx_file = self.server_file + self.ID + '_' + str(self.modelVersion+1) + "_lab_ind.pkl"
                model_path_head = self.server_file + self.ID + '_' + str(self.modelVersion+1)
                model_path = self.server_file + self.ID + '_' + str(self.modelVersion+1) + '-1.pth.tar'
                acc_path = self.server_file + self.ID + '_' + str(self.modelVersion+1) + '.txt'
                with open(idx_file, 'wb') as f:
                    pickle.dump(self.Pool_M.label_ind, f)
                print("start training........")
                os.system("python server_train.py " + idx_file + " " + model_path_head + " " + acc_path)
                print("......")

                while not os.path.exists(model_path):
                    print("server waiting........")
                    time.sleep(10)

                print(colored(
                    'end train---------' + str(self.ID) + " ,modelVersion: " + str(
                        self.modelVersion), "red"))

                self.Pool_M.latest_model_version += 1
                self.Pool_M.latest_mv_server = self.ID
                self.Pool_M.latest_model_path = model_path

                self.modelVersion += 1
                self.update = False

