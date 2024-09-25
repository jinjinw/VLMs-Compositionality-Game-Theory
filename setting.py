import torch
import numpy as np


class ExpSettings(object):
    """Experiment Setting Manager."""
    inference_benchmarks = [
        # "VG relation"
        {
        "benchmark_name": "VG_relation",
        "benchmark_pth": "./data/VG_relation",
        "annotation_file": "./visual_genome_relation.json",
        "preprocess_file": "./preprocess/VG/VG_relation_clip.py",
        "kwargs" : {
            'dataset_dir': "./data/VG_relation/images/", 
            'context_length': 77, "image_size": 224, 
        },
        }
        ]


exp_setting = ExpSettings()