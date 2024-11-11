import threading
from termcolor import colored
import numpy as np
import queue
import time
import os



class Pool_Manager(threading.Thread):
    def __init__(self, ID, train_set, test_set):
        threading.Thread.__init__(self)
        self.ID = ID
        self.TrainPool = train_set
        self.TestPool = test_set
        self.label_ind = [i for i, x in enumerate(train_set.unlabeled_mask) if x == 0]

        self.running = True
        self.que = queue.Queue()  ## 维护一个队列用来将查询的样本索引加入到索引集合中
        self.query_sample_num = 0

        # 用于worker模型更新
        self.latest_model_version = 0
        self.latest_mv_server = -1
        self.latest_model_path = ''

        # create the results path:
        self.filename = time.strftime("%Y-%m-%d-%H-%M-%S")
        os.mkdir(self.filename)
        os.mkdir(os.path.join(self.filename, 'server'))
        os.mkdir(os.path.join(self.filename, 'worker'))


    # overrides run
    def run(self):
        print(colored('开始线程: ' + self.ID, "white"))
        self.listen()
        print(colored('结束线程: ' + self.ID, "white"))

    def submitLabelData(self, query_index):
        self.label_ind.extend(query_index)

    # 监听函数
    def listen(self):
        while self.running:
            time.sleep(1)





