
import os
import torch
# from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from glob import glob
import cv2
import numpy as np
import itertools

import json
# from lightning_modules.data_modules.vie_dataset import VIEDataset
# from model import get_model
# from omegaconf import OmegaConf
from utils import get_class_names, get_config, get_label_map

import os 
import json

# from google.cloud import vision
from base64 import b64encode
import argparse
# from transformers import BertTokenizer
import numpy as np
from math import dist
import cv2
from PIL import Image
import shutil
# import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
import numpy as np




def get_eval_kwargs_geolayoutlm_vie(geo_clsses_path):
    print(geo_clsses_path)
    class_names = get_class_names(geo_clsses_path)
    bio_class_names = ["O"]
    for class_name in class_names:
        # if not class_name.startswith('O'):
        if class_name.upper() != 'O':
            bio_class_names.extend([f"B-{class_name}", f"I-{class_name}"])
    eval_kwargs = {
        "bio_class_names": bio_class_names,
    }
    return eval_kwargs





def getitem_geo(image, json_obj, tokenizer, backbone_type):
    return_dict = {}
    
    # class_names = ['data_cell', 'header_cell', 'trash', 'O']

    width = json_obj["meta"]["imageSize"]["width"]
    height = json_obj["meta"]["imageSize"]["height"]

    # img_path = os.path.join(dataset_root_path, json_obj["meta"]["image_path"])
    img_h=768
    img_w=768
    max_seq_length=512
    max_block_num=256
    
    # image = np.asarray(image)
    # image = cv2.resize(image, (img_w, img_h))
    # image = image.astype("float32").transpose(2, 0, 1)
    

    img_path = json_obj["meta"]["image_path"]

    image_ = cv2.resize(cv2.imread(img_path, 1), (img_w,img_h))
    image_ = image_.astype("float32").transpose(2, 0, 1)

    
    
    
    if getattr(tokenizer, "vocab", None) is not None:
        pad_token_id = tokenizer.vocab["[PAD]"]
        cls_token_id = tokenizer.vocab["[CLS]"]
        sep_token_id = tokenizer.vocab["[SEP]"]
        unk_token_id = tokenizer.vocab["[UNK]"]
    else:
        pad_token_id = tokenizer.pad_token_id
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        unk_token_id = tokenizer.unk_token_id

    # return_dict["image_path"] = img_path
    return_dict["image"] = image_
    return_dict["size_raw"] = np.array([width, height])
    
    return_dict["image_path"] = json_obj["meta"]["image_path"]
    return_dict["input_ids"] = np.ones(max_seq_length, dtype=int) * pad_token_id
    return_dict["bbox_4p_normalized"] = np.zeros((max_seq_length, 8), dtype=np.float32)
    return_dict["attention_mask"] = np.zeros(max_seq_length, dtype=int)
    return_dict["first_token_idxes"] = np.zeros(max_block_num, dtype=int)
    return_dict["block_mask"] = np.zeros(max_block_num, dtype=int)
    return_dict["bbox"] = np.zeros((max_seq_length, 4), dtype=np.float32)
    return_dict["line_rank_id"] = np.zeros(max_seq_length, dtype="int32")
    return_dict["line_rank_inner_id"] = np.ones(max_seq_length, dtype="int32")

    return_dict["are_box_first_tokens"] = np.zeros(max_seq_length, dtype=np.bool_)
    return_dict["bio_labels"] = np.zeros(max_seq_length, dtype=int)
    return_dict["el_labels_seq"] = np.zeros((max_seq_length, max_seq_length), dtype=np.float32)
    return_dict["el_label_seq_mask"] = np.zeros((max_seq_length, max_seq_length), dtype=np.float32)
    return_dict["el_labels_blk"] = np.zeros((max_block_num, max_block_num), dtype=np.float32)
    return_dict["el_label_blk_mask"] = np.zeros((max_block_num, max_block_num), dtype=np.float32)

    list_tokens = []
    list_bbs = [] # word boxes
    list_blk_bbs = [] # block boxes
    box2token_span_map = []

    box_to_token_indices = []
    cum_token_idx = 0

    cls_bbs = [0.0] * 8
    cls_bbs_blk = [0] * 4

    for word_idx, word in enumerate(json_obj["words"]):
        this_box_token_indices = []

        tokens = word["tokens"]
        bb = word["boundingBox"]
        if len(tokens) == 0:
            tokens.append(unk_token_id)

        if len(list_tokens) + len(tokens) > max_seq_length - 2:
            break # truncation for long documents

        box2token_span_map.append(
            [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
        )  # including st_idx, start from 1
        list_tokens += tokens

        # min, max clipping
        for coord_idx in range(4):
            bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
            bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

        bb = list(itertools.chain(*bb))
        bbs = [bb for _ in range(len(tokens))]

        for _ in tokens:
            cum_token_idx += 1
            this_box_token_indices.append(cum_token_idx) # start from 1

        list_bbs.extend(bbs)
        box_to_token_indices.append(this_box_token_indices)

    sep_bbs = [width, height] * 4
    sep_bbs_blk = [width, height] * 2

    first_token_idx_list = json_obj['blocks']['first_token_idx_list'][:max_block_num]
    if len(first_token_idx_list):
        if first_token_idx_list[-1] > len(list_tokens):
            blk_length = max_block_num
            for blk_id, first_token_idx in enumerate(first_token_idx_list):
                if first_token_idx > len(list_tokens):
                    blk_length = blk_id
                    break
            first_token_idx_list = first_token_idx_list[:blk_length]
        
    first_token_ext = first_token_idx_list + [len(list_tokens) + 1]
    line_id = 1
    for blk_idx in range(len(first_token_ext) - 1):
        token_span = first_token_ext[blk_idx+1] - first_token_ext[blk_idx]
        # block box
        bb_blk = json_obj['blocks']['boxes'][blk_idx]
        bb_blk[0] = max(0, min(bb_blk[0], width))
        bb_blk[1] = max(0, min(bb_blk[1], height))
        bb_blk[2] = max(0, min(bb_blk[2], width))
        bb_blk[3] = max(0, min(bb_blk[3], height))
        list_blk_bbs.extend([bb_blk for _ in range(token_span)])
        # line_rank_id
        return_dict["line_rank_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = line_id
        line_id += 1
        # line_rank_inner_id
        if token_span > 1:
            return_dict["line_rank_inner_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = [1] + [2] * (token_span - 2) + [3]

    # For [CLS] and [SEP]
    list_tokens = (
        [cls_token_id]
        + list_tokens[: max_seq_length - 2]
        + [sep_token_id]
    )
    if len(list_bbs) == 0:
        # When len(json_obj["words"]) == 0 (no OCR result)
        list_bbs = [cls_bbs] + [sep_bbs]
        list_blk_bbs = [cls_bbs_blk] + [sep_bbs_blk]
    else:  # len(list_bbs) > 0
        list_bbs = [cls_bbs] + list_bbs[: max_seq_length - 2] + [sep_bbs]
        list_blk_bbs = [cls_bbs_blk] + list_blk_bbs[: max_seq_length - 2] + [sep_bbs_blk]

    len_list_tokens = len(list_tokens)
    len_blocks = len(first_token_idx_list)
    return_dict["input_ids"][:len_list_tokens] = list_tokens
    return_dict["attention_mask"][:len_list_tokens] = 1
    return_dict["first_token_idxes"][:len(first_token_idx_list)] = first_token_idx_list
    return_dict["block_mask"][:len_blocks] = 1
    return_dict["line_rank_inner_id"] = return_dict["line_rank_inner_id"] * return_dict["attention_mask"]

    bbox_4p_normalized = return_dict["bbox_4p_normalized"]
    bbox_4p_normalized[:len_list_tokens, :] = list_bbs

    # bounding box normalization -> [0, 1]
    bbox_4p_normalized[:, [0, 2, 4, 6]] = bbox_4p_normalized[:, [0, 2, 4, 6]] / width
    bbox_4p_normalized[:, [1, 3, 5, 7]] = bbox_4p_normalized[:, [1, 3, 5, 7]] / height

    if backbone_type == "layoutlm":
        bbox_4p_normalized = bbox_4p_normalized[:, [0, 1, 4, 5]]
        bbox_4p_normalized = bbox_4p_normalized * 1000
        bbox_4p_normalized = bbox_4p_normalized.astype(int)

    return_dict["bbox_4p_normalized"] = bbox_4p_normalized
    bbox = return_dict["bbox"]

    bbox[:len_list_tokens, :] = list_blk_bbs
    # bbox -> [0, 1000)
    bbox[:, [0, 2]] = bbox[:, [0, 2]] / width * 1000
    bbox[:, [1, 3]] = bbox[:, [1, 3]] / height * 1000
    bbox = bbox.astype(int)
    return_dict["bbox"] = bbox

    st_indices = [
        indices[0]
        for indices in box_to_token_indices
        if indices[0] < max_seq_length
    ]
    return_dict["are_box_first_tokens"][st_indices] = True
    ########################################################
    return_dict["bio_labels"][0] = -100
    return_dict["bio_labels"][len_list_tokens:] = -100
    ########################################################
    
    for k in return_dict.keys():
        if isinstance(return_dict[k], np.ndarray):
            return_dict[k] = torch.from_numpy(return_dict[k])

    return return_dict