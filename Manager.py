import threading
from termcolor import colored
import time
import numpy as np

from datetime import datetime


class Manager(threading.Thread):
    def __init__(self, ID, Pool_M, server_dict, worker_dict, train_num, query_size):
        threading.Thread.__init__(self)
        self.ID = ID
        self.Pool_M = Pool_M
        self.server_dict = server_dict
        self.worker_dict = worker_dict
        self.train_num = train_num                # Server需要更新的总次数
        self.threshold = query_size*len(worker_dict)    # 结合increment进行理解，阈值是针对增量而言的
        self.train_iter = 0                                 # Server更新的总次数

        self.Size = len(self.Pool_M.label_ind)
        self.increment = 0

        self.index = 0 

    # overrides run
    def run(self):
        print(colored('开始线程: ' + self.ID, "yellow"))
        self.listen()
        print(colored('结束线程: ' + self.ID, "yellow"))

    # 监听函数
    def listen(self):
        time.sleep(5)
        while True:
            self.manage()
            if self.train_iter >= self.train_num:
                for i in range(len(self.server_dict)):
                    server_name = 'server-' + str(i + 1)
                    while self.server_dict[server_name].update:
                        time.sleep(1)
                    self.server_dict[server_name].running = False
                break

        self.Pool_M.running = False

    def manage(self):

        self.increment = len(self.Pool_M.label_ind) - self.Size
        if self.increment >= self.threshold:
            self.Size = len(self.Pool_M.label_ind)

            server_name = 'server-' + str((self.index - 1) % len(self.server_dict) + 1)
            while self.server_dict[server_name].update:
                time.sleep(1)

            self.server_dict[server_name].update = True
            print(colored('发送更新: ' + server_name, "yellow"))

            self.index = (self.index + 1) % len(self.server_dict)
            self.train_iter += 1

        else:
            time.sleep(1)

