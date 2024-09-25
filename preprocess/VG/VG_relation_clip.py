import os
from PIL import Image
import json
import hashlib

import torch
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from datafactory.data_utils.data_loader import BaseDatasetLoader
from datafactory.data_utils.data_process import DataProcess

import clip
torch.set_num_threads(1)


def create_id(item):
    m = hashlib.md5(str(item).encode("utf-8"))
    return m.hexdigest()


def contains_subarray(arr, subarr):
    n = subarr.shape[0]
    for i in range(arr.shape[0] - n + 1):
        if torch.equal(arr[i:i+n], subarr):
            return i
    return -1


def generate_mask(text, relation, mtype='true'):
    assert mtype in ['true', 'false']
    len_c = (text > 0).sum() - 2
    len_r = (relation > 0).sum() - 2
    start = torch.where(text == relation[1])[0]
    assert torch.equal(text[start:start+len_r], relation[1:1+len_r])
    attribute_mask = torch.zeros_like(text)
    relation_mask = torch.zeros_like(text)
    relation_mask[start-1:start+len_r] = 1 # to include "is"
    object1_mask = torch.zeros_like(text)
    object1_mask[1:start-1] = 1
    object2_mask = torch.zeros_like(text)
    object2_mask[start+len_r:1+len_c] = 1
    object_mask = object1_mask | object2_mask
    default_mask = torch.zeros_like(text)
    default_mask[0] = 1
    default_mask[len_c + 1] = 1
    return {f"inputs_object_mask_{mtype}": object_mask,
            f"inputs_relation_mask_{mtype}": relation_mask, 
            f"inputs_attribute_mask_{mtype}": attribute_mask, 
            f"inputs_default_mask_{mtype}": default_mask}


class ModelDataProcess(DataProcess):
    def __init__(
        self, filelist_pth, scorefile_path, image_size=224, 
        dataset_dir=None, vocab_file=None, context_length=None, **kwargs) -> None:
        super().__init__(filelist_pth)
        
        self.image_size = image_size
        self.dataset_dir = dataset_dir

        self.vocab_file = vocab_file
        self.scorefile_path = scorefile_path
        self.context_length = context_length
        
        def _convert_image_to_rgb(image):
            return image.convert("RGB")
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=BICUBIC),
            transforms.CenterCrop(image_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
            
    def preprocess(self, data_meta):
        try:
            # parse data meta infor.
            metas = data_meta
            meta_id = create_id(data_meta)
            image_pth = data_meta["image_path"]
            meta_text_true = data_meta["true_caption"]
            meta_text_false = data_meta["false_caption"]
            relation_name = data_meta["relation_name"]
            meta_label = 1
            if self.dataset_dir is not None:
                image_pth = os.path.join(self.dataset_dir, image_pth)
            
            processed_data = {'meta_text_true': meta_text_true, 
                              'meta_text_false': meta_text_false, 
                              'meta_image_pth':image_pth, 
                              'meta_relation_name': relation_name, 
                              'meta_label': meta_label, 
                              'meta_id': meta_id}
        except Exception as err:
            return None, f"[MetaSplitErr]: {err}, {data_meta}"
            
        try:
            image_data = Image.open(image_pth).convert("RGB")
            image_data = image_data.crop(
                (metas["bbox_x"], metas["bbox_y"], 
                 metas["bbox_x"] + metas["bbox_w"], 
                 metas["bbox_y"] + metas["bbox_h"])
            )
            image_data = self.transform(image_data)

            text_true = clip.tokenize(meta_text_true, context_length=self.context_length)[0]
            text_false = clip.tokenize(meta_text_false, context_length=self.context_length)[0]
            
            processed_data.update({
                'inputs_data': image_data, 
                'inputs_text_true': text_true, 
                'inputs_text_false': text_false})
            
            # generate mask in a stupid way
            text_relation = clip.tokenize(relation_name, context_length=self.context_length)[0]
            mask_true = generate_mask(text_true, text_relation, mtype='true')
            mask_false = generate_mask(text_false, text_relation, mtype='false')
            processed_data.update(mask_true)
            processed_data.update(mask_false)
            
            return processed_data, None
        except Exception as errmsg:
            return None, f"[PreprocessErr]: {errmsg}, {image_pth}."

    def load_dataset(self,):
        todo_atoms = list()
        with open(self.root_folder, "r") as f:
            atoms = json.load(f)
        if os.path.exists(self.scorefile_path):
            with open(self.scorefile_path, "r") as f:
                finished_atoms = json.load(f)
        else:
            finished_atoms = dict({"meta_id": list()})
        for atom in atoms:
            meta_id = create_id(atom)
            if meta_id not in finished_atoms["meta_id"]:
                todo_atoms.append(atom)
        return todo_atoms


if __name__ == '__main__':
    pass