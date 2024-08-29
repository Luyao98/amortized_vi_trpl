from toy_task.GMM.algorithms.algorithm_model2 import toy_task
from toy_task.GMM.algorithms.algorithm_direct import toy_task_2
from toy_task.GMM.algorithms.algorithm_stl import toy_task_3

def toy_task_algo(algo, config_dict):
    if algo==1:
        toy_task(config_dict)
    elif algo==2:
        toy_task_2(config_dict)
    elif algo==3:
        toy_task_3(config_dict)
    else:
        print("Invalid algorithm number")
        return
