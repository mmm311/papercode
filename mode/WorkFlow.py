from mode.Task import Task
import random
import config.constant as constant


class WorkFlow(object):
    def __init__(self):
        self.task_list = []
        self.make_span = None

    @property
    def task_list_length(self):
        return len(self.task_list)

    def get_task_by_id(self, _id):
        for task in self.task_list:
            if task.task_id == _id:
                return task

    @staticmethod
    def sort_dict_by_key(task_dict):
        # 相当于把dict的key的类型从string转换成int
        new_key_list = []
        old_key_list = []
        new_dict = {}
        for key in task_dict.keys():
            old_key_list.append(key)
            new_key_list.append(int(key))

        key_list_len = len(new_key_list)
        for i in range(key_list_len):
            # 先加后删
            task_dict[new_key_list[i]] = task_dict[old_key_list[i]]
            task_dict.pop(old_key_list[i])

        while len(new_key_list) >= 1:
            # 按照key的值进行排序
            key_min = min(new_key_list)
            new_dict[key_min] = task_dict[key_min]
            new_key_list.remove(key_min)

        task_dict.clear()
        task_dict = new_dict
        return task_dict

    def create_random_wf(self, task_number):
        entry_task = Task(0, [], [])
        exit_task = Task(task_number - 1, [], [])

        self.task_list.append(entry_task)
        self.task_list.append(exit_task)

        surplus_task_num = task_number - 2
        last_layer = [entry_task.task_id]
        max_id = 0

        while surplus_task_num != 0:
            current_layer_task_num = random.randint(constant.MIN_TASK_NUM, constant.MAX_TASK_NUM)

            # 创建当层任务列表
            current_layer_task_id_list = []

            if current_layer_task_num > surplus_task_num:
                current_layer_task_num = surplus_task_num

            for i in range(0, current_layer_task_num):
                task = Task(max_id + 1, [], [])
                self.task_list.append(task)
                current_layer_task_id_list.append(task.task_id)
                max_id += 1

            # 给上层任务随机添加后继任务
            for task_id in last_layer:
                task = self.get_task_by_id(task_id)

                random_num = random.randint(0, current_layer_task_num - 1)
                task_id_temp = current_layer_task_id_list[random_num]
                task.suc_task_id_list.append(task_id_temp)

                task_temp = self.get_task_by_id(task_id_temp)
                task_temp.pre_task_id_list.append(task_id)

            # 判断当前层的任务是否都有前驱，如果没有，随机添加前驱任务
            for task_id in current_layer_task_id_list:
                task = self.get_task_by_id(task_id)
                if task.pre_task_id_list is None or len(task.pre_task_id_list) == 0:
                    random_num = random.randint(0, len(last_layer) - 1)
                    task_id_temp = last_layer[random_num]
                    task_temp = self.get_task_by_id(task_id_temp)
                    task_temp.suc_task_id_list.append(task.task_id)
                    task.pre_task_id_list.append(task_id_temp)

            surplus_task_num -= current_layer_task_num
            last_layer = current_layer_task_id_list

        for task_id in last_layer:
            task = self.get_task_by_id(task_id)
            task.suc_task_id_list.append(exit_task.task_id)
            exit_task.pre_task_id_list.append(task_id)

    # 从文件中读取经典/固定的的工作流模型,只读取任务之间的先序关系
    def create_classic_wf(self, file_path):
        task_dict = dict()
        file = open(file_path, "r")
        line = file.readline()
        while line:
            # 首先去除换行符,真讨厌
            line = line.strip('\n')
            # 以空格分割一行,分割后的每一项都是task_id
            line_split = line.split(" ")

            # 如果当前task任务不在task_key_value中,那么new一个task
            for task_id in line_split:
                if task_id not in task_dict.keys():
                    task = Task(task_id, [], [])
                    task_dict[task_id] = task

            # 记录每一行的第一个task
            task = task_dict[line_split[0]]
            # 删除当前的task_id,那么剩余的line_split中的task都上它的后继
            line_split.pop(0)

            for task_id in line_split:
                task_temp = task_dict[task_id]
                task_temp.pre_task_id_list.append(task.task_id)
                task.suc_task_id_list.append(task_id)

            line = file.readline()

        task_dict = WorkFlow.sort_dict_by_key(task_dict)
        # 最后按照int类型的key值进行排序
        self.task_list = task_dict.values()