import os
import json
import glob
import builtins
import numpy as np
from PIL import Image
import math

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import scipy
import itertools
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from functools import lru_cache

from datafactory.utils import dataprocess_init
from datafactory.data_utils.data_loader import BaseDatasetLoader

from config import config
from infer_utils.model_compile import InferModelCompile


@lru_cache(maxsize=1)
def model_compile_init(local_rank, global_rank, world_size):
    return InferModelCompile(local_rank, global_rank, world_size)


def get_scorefile_name(scorefile_pth, model_name, global_rank):
    ckpt_name = os.path.basename(model_name)
    scorefile_pth = os.path.join(
        scorefile_pth, f"{ckpt_name}_{global_rank}.json"
    )

    if os.path.exists(scorefile_pth):
        return scorefile_pth, f"Loading Scorefile from {scorefile_pth}."
    return scorefile_pth, None


@torch.no_grad()
def run(
    local_rank, model_name, model_architecture, data_file, 
    dataprocess_file, scorefile_path, world_size, batch_size=128, 
    runner=4, kwargs=None):
    
    global_rank = local_rank

    if not os.path.exists(scorefile_path):
        os.makedirs(scorefile_path)

    scorefile_path, errmsg = get_scorefile_name(scorefile_path, model_name, global_rank)
    if errmsg is not None:
        print(errmsg)
        return 

    if global_rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    model = model_compile_init(local_rank, global_rank, world_size)
    
    print("DataLoader Init ing...")

    dataprocess = dataprocess_init(
        filelist_pth=data_file, dataprocess_file=dataprocess_file, scorefile_path=scorefile_path, **kwargs
    )
    dataloader = BaseDatasetLoader(
        dataprocess=dataprocess, global_rank=global_rank, world_size=world_size,
        batch_size=batch_size//world_size, runner=runner
    ).dataiter

    with torch.cuda.device(global_rank):
        print("Model Init ing...")
        model.model_init(None, model_architecture)

        print("Start Model Evaluation...")
        for data in dataloader:
            model.forward(data)
            model.postprocess(data)

        print("Dump Scorefile per batch...")
        model.dump_scorefile(scorefile_path)


def eval(
    model_name, model_architecture, data_file, preprocess_file, 
    scorefile_path, world_size, batch_size, runner_num, kwargs):
    
    _, errmsg = get_scorefile_name(scorefile_path, model_name, '0')
    if errmsg:
        print(errmsg)
        return 

    run(0, model_name, model_architecture, data_file, preprocess_file, 
        scorefile_path, world_size, batch_size, runner_num, kwargs)

    
def vis_results(scorefile_path):
    plt.rcParams.update({'font.size': 22})
    meta_ijk = {(1, 1, 0), (1, 0, 0), (0, 1, 0)}
    for meta_i, meta_j, meta_k in meta_ijk:
        acc_results = list()
        interaction_results = list()
        interaction_change_results = list()
        for json_file in glob.glob(scorefile_path):
            with open(json_file, 'r') as f:
                metas = json.load(f)
        for interaction, interaction_change, score in zip(metas[f'relation_s_{meta_i}_{meta_j}_{meta_k}'], metas[f'relation_s_change_{meta_i}_{meta_j}_{meta_k}'], metas['score']):
            acc_results.append(score)
            interaction_results.append(interaction)
            interaction_change_results.append(interaction_change)
        y = np.array(interaction_change_results)
        y_metric = np.array(interaction_results)
        metric = np.mean(y_metric)
        if meta_i == 1 and  meta_j == 1 and  meta_k == 0:
            print(f"Q_R&O: {metric}")
        if meta_i == 0 and  meta_j == 1 and  meta_k == 0:
            print(f"Q_R: {metric}")
        if meta_i == 1 and  meta_j == 0 and  meta_k == 0:
            print(f"Q_O: {metric}")
        x = np.array(acc_results)
        rho = scipy.stats.pearsonr(x, y)[0]
        print(f'rho:{rho}')
        sns.set_style("darkgrid")
        plt.subplots_adjust(left=0.15, right=0.85, top=0.93, bottom=0.1)
        wide_df = pd.DataFrame({"x": x, "y": y})
        sns.regplot(x="x", y="y", data=wide_df, 
                    color='r', ci=95, label='CLIP', marker='o', 
                    scatter_kws={'s':0.1}, line_kws={'linewidth':1}
                   ).set_title(f"Correlation:{format(rho, '.2f')}", weight='bold').set_fontsize('22')
        plt.xlabel("Reward Differences", fontweight='bold')
        if meta_i == 1 and  meta_j == 1 and  meta_k == 0:
            plt.ylabel("${\mathcal{Y}^\mathcal{T}_{R & O}}$ Differences", fontweight='bold')
        if meta_i == 0 and  meta_j == 1 and  meta_k == 0:
            plt.ylabel("${\mathcal{Y}^\mathcal{T}_R}$ Differences", fontweight='bold')
        if meta_i == 1 and  meta_j == 0 and  meta_k == 0:
            plt.ylabel("${\mathcal{Y}^\mathcal{T}_O}$ Differences", fontweight='bold')
        plt.legend(prop={'weight': 'bold'})
        plt.savefig(f'./interaction_{meta_i}_{meta_j}_{meta_k}.png', bbox_inches='tight')
        plt.clf()
        plt.cla()

        
def checkpoints_infer(model_name):
    from setting import exp_setting
    from config import config
    scorefile = os.path.join(config.COMM.EXP_TRAIN_LOG, "scorefile/")

    for bmk in exp_setting.inference_benchmarks:
        data_file = os.path.join(bmk['benchmark_pth'], bmk['annotation_file'])
        benchmark_name = bmk['benchmark_name']
        print("="* 10 + f"Start Inference {benchmark_name}" + "="* 10)
        scorefile_path = os.path.join(scorefile, benchmark_name)
        eval(
            model_name=model_name, model_architecture='model.py', data_file=data_file, 
            preprocess_file=bmk['preprocess_file'], scorefile_path=scorefile_path, 
            world_size=1, batch_size=128, runner_num=8, kwargs=bmk['kwargs']
        )
        scorefile_name, _ = get_scorefile_name(scorefile_path, model_name, 0)
        vis_results(scorefile_name)


if __name__ == "__main__":
    model_name = "clip"
    checkpoints_infer(model_name)