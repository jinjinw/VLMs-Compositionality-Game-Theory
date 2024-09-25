import os
import sys
import heapq
import importlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def model_init(model_weight, model_architecture=None, local_rank=0):
    '''
    该方法动态 import 外界 python file, 强制要求 model.py 中必须有 get_net() func.
    '''
    if not model_architecture:
        model_architecture = os.path.join(os.getcwd(), 'model.py')
    spec = importlib.util.spec_from_file_location(
        "module.name", model_architecture
    )
    model = importlib.util.module_from_spec(spec)
    sys.path.append(os.path.dirname(model_architecture))

    spec.loader.exec_module(model)
    try:
        net = model.get_net()
    except Exception:
        net = model.get_model()

    if model_weight is not None:
        model_state = dict()
        ckpt_state = torch.load(model_weight)
        ckpt_state = ckpt_state['model'] if 'model' in ckpt_state else ckpt_state

        for k, v in ckpt_state.items():
            model_state[k.replace('module.', '')] = v
        net.load_state_dict(model_state)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        net = net.cuda()
    net.eval()

    return net


@lru_cache(maxsize=1)
def dataprocess_init(filelist_pth=None, dataprocess_file=None, scorefile_path=None, **kwargs):
    '''
    该方法动态 import 外界 python file, 并实例化 ModelDataProcess 子类对象，
    用于完成数据加载 & 数据预处理的任务, 此方法强制要求 dataprocess.py 中必须有
    preprocess() func & load_dataset() func.
    '''
    if not dataprocess_file:
        dataprocess_file = os.path.join(os.getcwd(), 'dataprocess.py')

    spec = importlib.util.spec_from_file_location(
        "module.name", dataprocess_file
    )
    dataloader = importlib.util.module_from_spec(spec)
    sys.path.append(os.path.dirname(dataprocess_file))

    spec.loader.exec_module(dataloader)
    dataprocess = dataloader.ModelDataProcess(filelist_pth, scorefile_path, **kwargs)
    return dataprocess