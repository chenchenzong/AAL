import matplotlib
matplotlib.use('Agg')
import threading
from termcolor import colored
import time
import numpy as np
from datetime import datetime
import queue

date = datetime.now().strftime('%Y%m%d_%H%M%S')


class Pool_Manager(threading.Thread):
    def __init__(self, ID, TrainPool, TestPool, labeled_idx, unlabeled_idx):
        threading.Thread.__init__(self)
        self.ID = ID
        self.TrainPool = TrainPool
        self.TestPool = TestPool
        self.label_ind = labeled_idx
        self.unlab_ind = unlabeled_idx

        self.stop = False
        self.que = queue.Queue()  ## 维护一个队列用来将查询的样本索引加入到索引集合中
        self.query_num = 0

    # overrides run
    def run(self):
        print(colored('开始线程: ' + self.ID, "white"))
        self.listen()
        print(colored('结束线程: ' + self.ID, "white"))

    def submitLabelData(self, query_index):
        self.que.put(query_index)

    # 监听函数
    def listen(self):
        while self.stop == False:
            while self.que.empty() != True:
                self.label_ind = np.hstack((self.label_ind, self.que.get()))
                self.query_num += 1





