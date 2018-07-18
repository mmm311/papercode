'''
@author liu
@date 2018-07-05
@ PSO实现
'''
import numpy as np
import random
from config import constant
import math
import pickle
import os
from mode.WorkFlow import *


class PSO():
    def __init__(self, task_num, pop = 40, max_iter = 100, risk = 0):
        self.w = 0.95
        self.c1 = 2.15
        self.c2 = 2.05
        self.risk = risk
        self.server_info = [{'cores': 4, 'freq': 2.3, 'excutor':  2.1},
                            {"cores": 4, 'freq': 3.4, 'excutor': 3},
                            {'cores': 6, 'freq': 2.4,'excutor': 4},
                            {'cores': 8, 'freq': 2.1, 'excutor': 6}]

        # 加密速度
        self.encrypts = [11.76, 13.83, 22.03, 20.87, 37.17]
        self.integrity = [75.76, 101.01, 109.89, 119.05, 172.41]
        # 服务器 运行时长
        self.ss = []
        # 任务字典
        self.task_dict = None
        self.pop = pop #粒子数量
        self.task_num = task_num # 维度
        self.dim = self.task_num * 3 # 3个粒子一个任务
        self.max_iter = max_iter # 迭代次数
        # 工作流
        self.task_list = None
        # 任务顺序编码
        self.X_task = np.random.random((1, self.task_num))
        self.max_v = 2
        self.min_v = -2
        # 粒子群
        self.X = np.random.random((self.pop, self.dim))
        self.V = np.zeros((self.pop, self.dim)) # 所有粒子的速度
        # 粒子群的适应度
        self.y = np.zeros(self.pop)
        self.y_limit = np.zeros(self.pop)
        self.pbest_x = np.zeros((self.pop, self.dim)) # 个体经历的最佳位置
        self.pbest_y = np.zeros((2, self.pop)) # 个体历史最优解
        self.gbest_x = np.zeros((1, self.dim)) # 全局最佳位置
        self.gbest_y = np.array([1, 1]) * 1000 # 全局最佳
        self.gbest_y_hist = []
        self.gbest_yy_hist = []

    def cal_y(self):
        # 计算y值
        y = []
        for i, pop in enumerate(self.X):
            y.append(self.object_fun(self.task_dict, pop))
        self.y = y

    # 判断前继节点是否已经执行
    # 改
    @staticmethod
    def pre_task_finished(task_id, task_dict, encode):
        task = task_dict.get(task_id)
        if len(task.pre_task_id_list) == 0:
            return True
        for pre_task in task.pre_task_id_list:
            if pre_task not in encode:
                return False
        return True

    # 编码
    def encoding(self, tasks, task_dict):
        encode = []
        # 需要添加的任务
        task_append = []
        task_append.append(tasks[0].task_id)
        for task_id in task_append:
            if task_id not in encode and self.pre_task_finished(task_id, task_dict, encode):
                encode.append(task_id)
            task = task_dict.get(task_id)
            suc_id_list = task.suc_task_id_list

            if len(suc_id_list) == 0:
                break
            encode_list = []
            for suc_id in suc_id_list:
                if suc_id not in encode:
                    task_append.append(suc_id)
                    encode_list.append(suc_id)
            random.shuffle(encode_list)
            encode.extend(encode_list)

        return encode

    # 计算加密解密时间
    def cal_encrypt(self, server, algo_index, data):
        cores, freq = self.server_info[server].get('cores'), self.server_info[server].get("freq")
        return data / (self.encrypts[algo_index] * cores * freq / 2.2)

    # 计算完整时间
    def cal_integrity(self, pre_server, server, algo_index, data):

        cores, freq = self.server_info[server].get('cores'), self.server_info[server].get('freq')
        if pre_server == 0:
            return data / (self.integrity[algo_index] * cores * freq / 2.2 )
        else:
            pre_server_cores = self.server_info[pre_server].get('cores')
            return math.ceil(pre_server_cores / cores) * data / (self.integrity[algo_index] * cores * freq / 2.2 )

    def init_workflow(self):
        self.task_dict = {}
        encode = []
        if not os.path.exists("task"+str(self.task_num)+".pkl"):
            workflow = WorkFlow()
            workflow.create_random_wf(self.task_num)
            self.task_list = workflow.task_list

            for task in self.task_list:
                self.task_dict[task.task_id] = task
                if task.task_id != 0 and task.task_id != self.task_num - 1:
                    task.work_load = random.random() * (constant.MAX_WORKFLOW - constant.MIN_WORKFLOW) + constant.MIN_WORKFLOW
                task.output_data = random.random() * (constant.MAX_OUTPUT - constant.MIN_OUTPUT) + constant.MIN_OUTPUT
            encode = self.encoding(self.task_list, self.task_dict)
            with open('task' + str(self.task_num) +'.pkl', 'wb') as f:
                pickle.dump([self.task_list, encode], f)
        else:
            self.task_dict = {}
            with open('task'+ str(self.task_num)+".pkl",'rb') as f:
                self.task_list, encode = pickle.load(f)
            for task in self.task_list:
                self.task_dict[task.task_id] = task

        for i in range(self.pop):
            pp = []
            for task_id in encode:
                server = random.randint(0, constant.SERVER_NUM)
                encrypt = random.randint(0, constant.ENCRYPT - 1)
                integrity = random.randint(0, constant.INTEGRITY - 1)
                pp.append(server)
                pp.append(encrypt)
                pp.append(integrity)
            self.X[i, :] = pp
        self.X_task = encode

    def cal_probility(self, algo, algo_index):
        probility = [[1.0, 0.85, 0.53, 0.56, 0.32], [1.0, 0.75, 0.69, 0.63, 0.44]]
        thredhold = [2.5, 1.8]
        return 1 - math.exp(-1 * thredhold[algo] * (1 - probility[algo][algo_index]))

    def object_fun(self, task_dict, particle):
        s = [0] * (constant.SERVER_NUM + 1)
        flag = [-1] * self.task_num
        E = 0
        # 初始化任务
        for task_id in task_dict:
            task_dict.get(task_id).start_time = 0
            task_dict.get(task_id).exc_time = 0
            task_dict.get(task_id).end_time = 0
            task_dict.get(task_id).P = 0

        for i, task_id in enumerate(self.X_task):
            task = task_dict.get(task_id)
            task.server = int(particle[i * 3])
            task.encrypt = int(particle[i * 3 + 1])
            task.integrity = int(particle[i * 3 + 2])

            if task_id == 0 or task_id == self.task_num - 1:
                task.server = 0

            pre_task_id_list = task.pre_task_id_list

            # 任务的前继任务索引为0
            if len(pre_task_id_list) == 0:
                task.start_time = 0

                task.end_time = 0
                s[task.server] = 0
            else:
                confi = 0
                for pre_task_id in pre_task_id_list:
                    pre_task = task_dict.get(pre_task_id)
                    # 判断是否在同一个服务器
                    if pre_task.server == task.server:
                         task.start_time =max(task.start_time, pre_task.end_time)
                    else:
                        if flag[pre_task.task_id] == -1:
                            security = self.cal_encrypt(pre_task.server,pre_task.encrypt, pre_task.output_data)+ \
                                                self.cal_integrity(0, pre_task.server, pre_task.integrity, pre_task.output_data)\
                                                + pre_task.output_data / constant.WIDTHBAND
                            pre_task.exc_time = pre_task.exc_time + security
                            pre_task.end_time = pre_task.start_time + pre_task.exc_time
                            s[pre_task.server] = s[pre_task.server] + security
                            pre_task.P = 1 - ((1 - self.cal_probility(0, pre_task.encrypt)) * (1 - self.cal_probility(1, pre_task.integrity)))
                            flag[pre_task.task_id] = 1

                        task_dict.get(pre_task.task_id).end_time = pre_task.end_time
                        task_dict.get(pre_task.task_id).P = pre_task.P
                        s[pre_task.server] = pre_task.end_time
                        task.start_time =max(pre_task.end_time, task.start_time)
                    confi = self.cal_encrypt(task.server, pre_task.encrypt, pre_task.output_data) + \
                            self.cal_integrity(pre_task.server, task.server, pre_task.integrity, pre_task.output_data) + confi
                executor_time = task.work_load / self.server_info[task.server].get('excutor') + confi

                task.exc_time = executor_time + task.exc_time
                task.end_time = task.start_time + task.exc_time
                s[task.server] = s[task.server] + executor_time
        end_time = task_dict.get(self.task_num - 1).end_time
        P = 0
        for task_id in task_dict:
            task = task_dict.get(task_id)
            if task.server == 0:
              E = E + task.exc_time * constant.ENERGY
            P = (1 - task.P) + P
        P = 1 - P / self.task_num

        limit = max(0, end_time - constant.DEADTIME) + max(0, P - self.risk)
        return limit, E

    def object_fun2(self, task_dict, particle):
        s = [0] * (constant.SERVER_NUM + 1)
        flag = [-1] * self.task_num
        E = 0
        # 初始化任务
        for task_id in task_dict:
            task = task_dict.get(task_id)
            task.start_time = 0
            task.exc_time = 0
            task.end_time = 0
            task.P = 0

        for i, task_id in enumerate(self.X_task):
            task = task_dict.get(task_id)
            task.server = int(particle[i * 3])
            task.encrypt = int(particle[i * 3 + 1])
            task.integrity = int(particle[i * 3 + 2])

            if task_id == 0 or task_id == self.task_num - 1:
                task.server = 0

            suc_task_id_list = task.suc_task_id_list

            for suc_task_id in suc_task_id_list:
                suc_task = task_dict.get(suc_task_id)
                if suc_task.server == task.server:
                    pass

    def init_population(self):
        V = np.random.random((self.pop, self.dim))
        self.init_workflow()
        self.V = V * 2
        self.constrain_XV()
        self.cal_y()
        self.pbest_y = self.y
        self.pbest_x = self.X
        for i in range(self.pop):
            if self.pbest_y[i][0] == 0 and self.gbest_y[0] == 0:
                if self.gbest_y[1] > self.pbest_y[i][1]:
                    self.gbest_x = self.pbest_x[i, :]
                    self.gbest_y = self.pbest_y[i]

            if self.pbest_y[i][0] == 0 and self.gbest_y[0] != 0:
                self.gbest_x = self.pbest_x[i, :]
                self.gbest_y = self.pbest_y[i]

            if self.y[i][0] != 0 and self.gbest_y[0] != 0:
                if self.y[i][0] < self.gbest_y[0]:
                    self.gbest_x = self.X[i, :]
                    self.gbest_y = self.y[i]
                elif self.y[i][0] == self.gbest_y[0] and self.gbest_y[1] > self.y[i][1]:
                    self.gbest_x = self.X[i, :]
                    self.gbest_y = self.y[i]

        self.pbest_x = self.X
        self.gbest_y_hist.append(self.gbest_y[1])

    def constrain_XV(self):
        # 越界控制
        for i in range(self.pop):
            for j in range(int(self.dim / 3)):
                # 服务器越界·
                if self.X[i][3 * j] >= constant.SERVER_NUM:
                    self.X[i][3 * j] = constant.SERVER_NUM
                    self.V[i][3 * j] = -1 * self.V[i][3 * j]
                elif self.X[i][3 * j] < 0:
                    self.X[i][3 * j] = 0
                    self.V[i][3 * j] = -1 * self.V[i][3 * j]
                else:
                    self.X[i][3 * j] = int(self.X[i][3 * j])

                # 机密性服务越界
                if self.X[i][3 * j + 1] > 4:
                    self.X[i][3 * j + 1] = 4
                    self.V[i][3 * j + 1] = -1 * self.V[i][3 * j + 1]
                elif self.X[i][3 * j + 1] < 0:
                    self.X[i][3 * j + 1] = 0
                    self.V[i][3 * j + 1] = -1 * self.V[i][3 * j + 1]
                else:
                    self.X[i][3 * j + 1] = int(self.X[i][3 * j + 1])

                # 完整性服务越界
                if self.X[i][3 * j + 2] > constant.INTEGRITY - 1:
                    self.X[i][3 * j + 2] = constant.INTEGRITY - 1
                    self.V[i][3 * j + 2] = -1 * self.V[i][3 * j + 2]
                elif self.X[i][3 * j + 2] < 0:
                    self.X [i][3 * j + 2] = 0
                    self.V[i][3 * j + 2] = -1 * self.V[i][3 * j + 2]
                else:
                    self.X[i][3 * j + 2] = int(self.X[i][3 * j + 2])
            self.X[i][0] = 0
            self.X[i][-3] = 0

        for i in range(self.pop):
            for j in range(self.dim):
                if self.V[i][j] > self.max_v:
                    self.V[i][j] = self.max_v

                if self.V[i][j] < self.min_v:
                    self.V[i][j] = self.min_v


    def fit(self):
        self.init_population()
        for j in range(self.max_iter):
            self.V = self.w * self.V + self.c1 * random.random() * (self.pbest_x - self.X) +\
                     self.c2 * random.random() * (self.gbest_x - self.X)

            # for ii in range(len(self.V)):
            #     for jj in range(len(self.V[0])):
            #         self.V[ii][jj] = int(self.V[ii][jj])
            self.X = self.X + self.V
            self.constrain_XV()
            #
            print("X", self.X)
            print("V", self.V)
            print("g - x", self.gbest_x - self.X)
            print('p - x ',self.pbest_x - self.X)
            self.cal_y()
            for i in range(self.pop):
                if self.y[i][0] == 0 and self.pbest_y[i][0] == 0:
                    if self.pbest_y[i][1] > self.y[i][1]:
                        self.pbest_x[i, :] = self.X[i, :]
                        self.pbest_y[i] = self.y[i]

                if self.y[i][0] == 0 and self.pbest_y[i][0] != 0:
                    self.pbest_x[i, :] = self.X[i, :]
                    self.pbest_y[i] = self.y[i]

                if self.y[i][0] > 0 and self.pbest_y[i][0] > 0:
                    if self.y[i][0] < self.pbest_y[i][0]:
                        self.pbest_x[i, :] = self.X[i, :]
                        self.pbest_y[i] = self.y[i]

                    elif self.y[i][0] == self.pbest_y[i][0] and self.pbest_y[i][1] > self.y[i][1]:
                        self.pbest_x[i, :] = self.X[i, :]
                        self.pbest_y[i] = self.y[i]

            for i in range(self.pop):
                if self.pbest_y[i][0] == 0 and self.gbest_y[0] == 0:
                    if self.gbest_y[1] > self.pbest_y[i][1]:
                        self.gbest_x = self.pbest_x[i, :]
                        self.gbest_y = self.pbest_y[i]

                if self.pbest_y[i][0] == 0 and self.gbest_y[0] != 0:
                    self.gbest_x = self.pbest_x[i, :]
                    self.gbest_y = self.pbest_y[i]

                if self.pbest_y[i][0] != 0 and self.gbest_y[0] != 0:
                    if self.pbest_y[i][0] < self.gbest_y[0]:
                        self.gbest_x = self.pbest_x[i, :]
                        self.gbest_y = self.pbest_y[i]
                    elif self.pbest_y[i][0] == self.gbest_y[0] and self.gbest_y[1] > self.pbest_y[i][1]:
                        self.gbest_x = self.pbest_x[i, :]
                        self.gbest_y = self.pbest_y[i]
            # print(j, self.gbest_y)
            self.gbest_yy_hist.append(self.gbest_y[0])
            self.gbest_y_hist.append(self.gbest_y[1])



