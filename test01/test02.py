from mode.PSO import *
import time
for task_num in [15]:
    for risk in [.5]:
        for i in range(1):
            start = time.time()
            pso = PSO(task_num, 1, 500, risk)
            pso.fit()
            end = time.time()
            print('次数:', i, task_num, risk, pso.gbest_y, end - start)
            task_list = pso.task_list









