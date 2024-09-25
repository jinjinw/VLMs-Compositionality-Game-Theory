import torch
import torch.nn as nn
import math
import os
import clip


def get_net() -> torch.nn.Module:
    root_dir = "./models/clip"
    model, _ = clip.load("ViT-B/32", download_root=root_dir)
    return model 


if __name__ == '__main__':
    model = get_net()