import matplotlib

matplotlib.use('Agg')
import threading
from termcolor import colored
import time
import numpy as np

from datetime import datetime


class Manager_Worker(threading.Thread):
    def __init__(self, ID, server_dict, severId_head, worker_dict, WorkerID_head, ServerTrainNum, ServerTrainThreshold):
        threading.Thread.__init__(self)
        self.ID = ID
        self.server_dict = server_dict
        self.serverId_head = severId_head
        self.worker_dict = worker_dict
        self.WorkerID_head = WorkerID_head
        self.ServerTrainNum = ServerTrainNum                # Server需要更新的总次数
        self.ServerTrainThreshold = ServerTrainThreshold    # 结合increment进行理解，阈值是针对增量而言的
        self.train_iter = 0                                 # Server更新的总次数
        
        # basesize是一次server更新后标签集的大小，size为不断查询后标签集的大小，两者的差值即增量用来控制何时驱动下一个Server
        self.Size = server_dict["Server1"].Pool_M.label_ind.shape[0]
        self.BaseSize = server_dict["Server1"].Pool_M.label_ind.shape[0]
        self.increment = self.Size - self.BaseSize

        #index用来索引不同的Server，采用流水线的思想，每隔一段时间会驱动下一个Server
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

            ######################### 停止所有的worker和server ####################################
            if self.train_iter >= self.ServerTrainNum:
                for i in range(len(self.server_dict)):
                    server_name = self.serverId_head + str(i + 1)
                    while self.server_dict[server_name].update == True:
                        time.sleep(0.1)
                    self.server_dict[server_name].stop = True
                for i in range(len(self.worker_dict)):
                    worker_name = self.WorkerID_head + str(i + 1)
                    self.worker_dict[worker_name].stop = True
                break

        ######################### 结果的保存 ####################################
        result_list = []
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        for i in range(len(self.server_dict)):
            server_name = self.serverId_head + str(i + 1)
            result_list.extend(self.server_dict[server_name].score_list)
        result_list = np.array(result_list).reshape(-1, 2)
        result_list = result_list[result_list[:, 0].argsort()]
        np.savetxt("result/" + date + "_scores.csv", result_list, fmt="%d %.4f")
        self.server_dict["Server1"].Pool_M.stop = True

    def manage(self):
        
        ### worker查询一定量后server进行更新
        self.increment = self.server_dict["Server1"].Pool_M.label_ind.shape[0] - self.Size
        if self.increment >= self.ServerTrainThreshold:
            self.Size = self.server_dict["Server1"].Pool_M.label_ind.shape[0]

			## 由于性能不足，一次只允许运行一个server，性能允许的情况下这里可注释掉
            server_name = self.serverId_head + str((self.index - 1) % len(self.server_dict) + 1)
            while self.server_dict[server_name].update == True:
                time.sleep(0.1)

            server_name = self.serverId_head + str(self.index + 1)

            ## 同一个server，上一次训练还没结束的话需要先等待
            while self.server_dict[server_name].update == True:
                time.sleep(0.1)
            self.server_dict[server_name].update = True
            print(colored('发送更新: ' + server_name, "yellow"))

            self.index = (self.index + 1) % len(self.server_dict)
            self.train_iter += 1

