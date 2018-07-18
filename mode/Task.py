
class Task():
   '''
   :param task_id :任务id
           pre_task_id_list: 前继任务id
           suc_task_id_list: 后继任务id
   '''
   def __init__(self, task_id, pre_task_id_list = [], suc_task_id_list = []):
        self.task_id = task_id
        self.pre_task_id_list = pre_task_id_list
        self.suc_task_id_list = suc_task_id_list
        self.work_load = 0
        self.input_data = 0
        self.output_data = 0
        # 执行服务器
        self.server = -1
        # 机密性
        self.encrypt = -1
        # 完整性
        self.integrity = -1

        self.exc_time = -1
        self.start_time = -1
        self.end_time = -1

        # 风险率
        self.P = 0