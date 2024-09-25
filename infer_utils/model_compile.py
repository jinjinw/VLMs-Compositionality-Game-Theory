import os 
import json
import itertools

import torch
from einops import rearrange

from collections import defaultdict
from datafactory.utils import model_init


class InferModelCompile(object):
    """Model Inference Object."""
    def __init__(self, local_rank, global_rank, world_size):
        
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.scorefile = defaultdict(list)

    def model_init(self, model_weight, model_architecture):
        """init model architecture and load pretrained model."""
        self.model = model_init(model_weight, model_architecture, local_rank=self.local_rank)

    def l2_norm(self, embed):
        return embed / (embed.norm(dim=-1, keepdim=True) + 1e-12)

    def forward(self, data):
        """inference function."""
        for k, v in data.items():
            if 'meta' in k:
                continue
            data[k] = torch.stack(v).cuda(non_blocking=True)
        
        query_embed = self.l2_norm(self.model.encode_image(data['inputs_data']))
        text_true = data['inputs_text_true']
        text_false = data['inputs_text_false']
        ori_gallery_gt_embed_ori = self.l2_norm(self.model.encode_text(text_true))
        ori_gallery_wr_embed_ori = self.l2_norm(self.model.encode_text(text_false))
        data['score'] = (ori_gallery_gt_embed_ori @ query_embed.t()).diag() - (ori_gallery_wr_embed_ori @ query_embed.t()).diag()
        meta_ijk = {(1, 1, 0), (1, 0, 0), (0, 1, 0)}
        z_d = 0
        z_s = 0
        for meta_i, meta_j, meta_k in meta_ijk:
            gallery_gt_embed_interaction = 0
            gallery_wr_embed_interaction = 0
            for i, j, k in itertools.product(range(0, meta_i+1), range(0, meta_j+1), range(0, meta_k+1)):
                text_true = data['inputs_text_true'] * (data['inputs_object_mask_true'] * i) + \
                data['inputs_text_true'] * (data['inputs_relation_mask_true'] * j) + \
                data['inputs_text_true'] * (data['inputs_attribute_mask_true'] * k) + \
                data['inputs_text_true'] * data['inputs_default_mask_true']
                gallery_gt_embed = self.l2_norm(self.model.encode_text(text_true))
                gallery_gt_embed = (gallery_gt_embed @ query_embed.t()).diag()
                gallery_gt_embed_interaction += gallery_gt_embed * ((-1) ** (meta_i + meta_j + meta_k - i - j - k))

                text_false = data['inputs_text_false'] * (data['inputs_object_mask_false'] * i) + \
                data['inputs_text_false'] * (data['inputs_relation_mask_false'] * j) + \
                data['inputs_text_false'] * (data['inputs_attribute_mask_false'] * k) + \
                data['inputs_text_false'] * data['inputs_default_mask_false']
                gallery_wr_embed = self.l2_norm(self.model.encode_text(text_false))
                gallery_wr_embed = (gallery_wr_embed @ query_embed.t()).diag()
                gallery_wr_embed_interaction += gallery_wr_embed * ((-1) ** (meta_i + meta_j + meta_k - i - j - k))
            
            data[f'interaction_gt_{meta_i}_{meta_j}_{meta_k}'] = gallery_gt_embed_interaction
            data[f'interaction_wr_{meta_i}_{meta_j}_{meta_k}'] = gallery_wr_embed_interaction
            z_d += gallery_gt_embed_interaction ** 2
            z_s += gallery_gt_embed_interaction ** 2
            z_s += gallery_wr_embed_interaction ** 2

        z_d = z_d / len(meta_ijk)
        z_s = z_s / (len(meta_ijk) * 2)
        
        for meta_i, meta_j, meta_k in meta_ijk:   
            data[f'relation_s_{meta_i}_{meta_j}_{meta_k}'] = ((data[f'interaction_gt_{meta_i}_{meta_j}_{meta_k}'] - data[f'interaction_wr_{meta_i}_{meta_j}_{meta_k}']) ** 2) / z_s
            data[f'relation_s_change_{meta_i}_{meta_j}_{meta_k}'] = data[f'interaction_gt_{meta_i}_{meta_j}_{meta_k}'] - data[f'interaction_wr_{meta_i}_{meta_j}_{meta_k}']

            accu = 0
            for meta_i_, meta_j_, meta_k_ in meta_ijk:
                accu += ((data[f'interaction_gt_{meta_i}_{meta_j}_{meta_k}'] - data[f'interaction_gt_{meta_i_}_{meta_j_}_{meta_k_}']) ** 2)
            
            data[f'relation_d_{meta_i}_{meta_j}_{meta_k}'] = accu / z_d
            
    def postprocess(self, data):
        """data postprocess function for dumping json scorefile."""
        for k, v in data.items():
            if 'inputs' in k:
                continue
            v = v if 'meta' in k else [vv.item() for vv in v]
            self.scorefile[k] += v
            
    def dump_scorefile(self, scorefile_name):
        """dump json socrefile."""
        with open(scorefile_name, 'w') as f:
            json.dump(self.scorefile, f)