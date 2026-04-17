import os
import torch
from tqdm import tqdm
from glob import glob
import cv2
import numpy as np
import itertools
from model import get_model
from omegaconf import OmegaConf
from utils import get_class_names, get_config, get_label_map
import json
# from google.cloud import vision
from base64 import b64encode
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from PIL import Image
import numpy as np
import configparser
import shutil
import json
from math import dist
from copy import copy
import os
import json
from functools import cmp_to_key
from tqdm import tqdm
from PIL import Image, ImageDraw
import shutil
from typing import Dict, List

from glob import glob

import imagesize
from transformers import BertTokenizer
from typing import Dict


from prediction_utility import result_generation

from result_utility import get_eval_kwargs_geolayoutlm_vie, getitem_geo
from prediction_utility import result_generation
from pre_process_utility import main, gv_data

from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request,Depends, Header
# from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import FastAPI,Request
from io import BytesIO
import base64




App_Filepath = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config.read(App_Filepath + '/config.ini')

geo_clsses_path = config['PATH']['GEO_CLASSES_PATH']
geo_dump_dir = config['PATH']['GEO_DUMP']
img_path = config['PATH']['DATA']




def load_model_weight(net, pretrained_model_file):
    print("Loading ckpt from:", pretrained_model_file)
    print("HERE")
    pretrained_model_state_dict = torch.load(pretrained_model_file, map_location="cpu")
    print("HERE 2")
    if "state_dict" in pretrained_model_state_dict.keys():
        pretrained_model_state_dict = pretrained_model_state_dict["state_dict"]
    new_state_dict = {}
    valid_keys = net.state_dict().keys()
    invalid_keys = []
    for k, v in pretrained_model_state_dict.items():
        new_k = k
        if new_k.startswith("net."):
            new_k = new_k[len("net.") :]
        if new_k in valid_keys:
            new_state_dict[new_k] = v
        else:
            invalid_keys.append(new_k)
    print(f"These keys are invalid in the ckpt: [{','.join(invalid_keys)}]")
    net.load_state_dict(new_state_dict)






from lightning_modules.geolayoutlm_vie_module import (
    do_eval_epoch_end,
    do_eval_step
)
eval_kwargs = get_eval_kwargs_geolayoutlm_vie(geo_clsses_path)





def load_model():
    mode = "val"
    geo_cfg = get_config('./configs/finetune_funsd.yaml')
    # geo_cfg.dump_dir = '/New_Volume/number_theory/GEO_Rakesh/master_table_extraction/data/output/invoice_test/headers/geo_out'
    pt_list = os.listdir(os.path.join(geo_cfg.workspace, "checkpoints"))
    if len(pt_list) == 0:
        print("Checkpoint file is NOT FOUND!")
    pt_to_be_loaded = pt_list[0]
    if len(pt_list) > 1:
        # import ipdb;ipdb.set_trace()
        for pt in pt_list:
            if geo_cfg[mode].pretrained_best_type in pt:
                pt_to_be_loaded = pt
                break
    geo_cfg.pretrained_model_file = os.path.join(geo_cfg.workspace, "checkpoints", pt_to_be_loaded)
    print(geo_cfg)
    
    net = get_model(geo_cfg)
    load_model_weight(net, geo_cfg.pretrained_model_file)
    net.to("cpu")
    net.eval()
    
    return net


def predict(net, image, json_obj, backbone_type='geolayoutlm'):
    tokenizer = net.tokenizer
    input_data = getitem_geo(image, json_obj, tokenizer, backbone_type)
    print('input_data >>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print(input_data)
    # net.to("cuda")
    # net.eval()
    #device = 'cpu'
    # model.to(device)
    device = next(net.parameters()).device
    for key, value in input_data.items():
    # Apply torch.unsqueeze to each element in the tensor
        try:
            input_data[key] = torch.unsqueeze(value, 0)
            input_data[key] = input_data[key].to(device)
        except:
            pass
    with torch.no_grad():
        # head_outputs, loss_dict = net(input_data)
        head_outputs, loss_dict = net(input_data)
        # output = net(torch.unsqueeze(input_data))
        # print(output)
    pr_labels = torch.argmax(head_outputs["logits4labeling"], -1)
    step_out, final_results = do_eval_step(input_data, head_outputs, loss_dict, eval_kwargs, dump_dir=geo_dump_dir)
    # print(step_out)
    # print(final_results)
    # step_outputs.append(step_out)
    # print(pr_labels)
    return pr_labels, final_results




def preprocessData(json_file:Dict, img_path, MAX_SEQ_LENGTH = 512, MODEL_TYPE = "bert", 
                    VOCA = "bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)
    
    
    
    ######################################################
    ######################################################
    
    
    # OUTPUT_PATH = os.path.join(input_path, "dataset/custom_geo")
    # os.makedirs(OUTPUT_PATH, exist_ok=True)
    # os.makedirs(os.path.join(OUTPUT_PATH, "preprocessed"), exist_ok=True)
    # if dataset_split == "train":
    #     dataset_root_path = os.path.join(input_path, "training_data")
    # elif dataset_split == "test":
    #     dataset_root_path = os.path.join(input_path, "testing_data")
    # elif dataset_split == "val":
    #     dataset_root_path = os.path.join(input_path, "validation_set")
    #     if not os.path.exists(dataset_root_path):
    #         os.makedirs(dataset_root_path)
    #     try:
    #         if not os.path.exists(os.path.join(dataset_root_path,'images')):
    #             shutil.copytree(os.path.join(input_path,'images'), os.path.join(dataset_root_path,'images'))
    #             shutil.copytree(os.path.join(input_path,'annotations'), os.path.join(dataset_root_path,'annotations'))
    #     except Exception as e:
    #         print(e)

    # else:
    #     raise ValueError(f"Invalid dataset_split={dataset_split}")
    
    
    ###################################################################################
    ###################################################################################
    ###################################################################################
    # json_files = glob(os.path.join(dataset_root_path, anno_dir, "*.json"))
    preprocessed_fnames = []
    # for json_file in tqdm(json_files):
    in_json_obj = json_file
    print('&*10')
    # print(in_json_obj)
    # exit('++++++++++++++++')

    out_json_obj = {}
    out_json_obj['blocks'] = {'first_token_idx_list': [], 'boxes': []}
    out_json_obj["words"] = []
    form_id_to_word_idx = {} # record the word index of the first word of each block, starting from 0
    other_seq_list = {}
    num_tokens = 0

    # words
    for form_idx, form in enumerate(in_json_obj):
        print(form)
        # exit('+++++++++++++++')
        form_id = form["id"]
        # form_text = form["text"].strip()
        # form_label = form["label"]
        # if form_label.startswith('O') or form_label.startswith('o'):
        #     form_label = "O"
        # form_linking = form["linking"]
        form_box = form["box"]

        # if len(form_text) == 0:
        #     continue # filter text blocks with empty text

        word_cnt = 0
        class_seq = []
        real_word_idx = 0
        for word_idx, word in enumerate(form["words"]):
            word_text = word["text"]

            print(f'word text: {word_text}')
            # exit('+++++++++++++++')
            bb = word["box"]
            # form_box=bb
            print(bb)
            # exit('++++++++++++++')
            bb = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))
            print(f'input id token:{tokens}')
            # exit('+++++++++++++')

            word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
            if len(word_text) != 0: # filter empty words
                out_json_obj["words"].append(word_obj)
                if real_word_idx == 0:
                    out_json_obj['blocks']['first_token_idx_list'].append(num_tokens + 1)
                num_tokens += len(tokens)

                word_cnt += 1
                class_seq.append(len(out_json_obj["words"]) - 1) # word index
                real_word_idx += 1
        if real_word_idx > 0:
            out_json_obj['blocks']['boxes'].append(form_box)
    # meta
    out_json_obj["meta"] = {}
    
    
    
    # image_file = (
    #     os.path.join(dataset_root_path,'images',file+ ".png")
    # )
    # print(image_file)
    # if not os.path.exists(image_file):       # if image has no ocr files currently skipping
    #     return True

    # # exit('+++++++++++++++')
    # if dataset_split == "train":
    #     out_json_obj["meta"]["image_path"] = image_file[
    #         image_file.find("training_data/") :
    #     ]
    # elif dataset_split == "test":
    #     out_json_obj["meta"]["image_path"] = image_file[
    #         image_file.find("testing_data/") :
    #     ]
    # elif dataset_split == "val":
    #     out_json_obj["meta"]["image_path"] = image_file[
    #         image_file.find("validation_set/") :
    #     ]
    out_json_obj["meta"]["image_path"] = img_path
    width, height = imagesize.get(img_path)
    out_json_obj["meta"]["imageSize"] = {"width": width, "height": height}
    out_json_obj["meta"]["voca"] = VOCA

    # this_file_name = file+'.json'

    # # # Save file name to list
    # preprocessed_fnames= os.path.join("preprocessed", this_file_name)
    # print(preprocessed_fnames)
    # # exit('+++++++++++++++++++')
    
    # # Save to file
    # data_obj_file = os.path.join(OUTPUT_PATH, "preprocessed", this_file_name)
    # with open(data_obj_file, "w", encoding="utf-8") as fp:
    #     json.dump(out_json_obj, fp, ensure_ascii=False)

    # # Save file name list file
    # preprocessed_filelist_file = os.path.join(
    #     OUTPUT_PATH, f"preprocessed_files_{dataset_split}.txt"
    # )
    # with open(preprocessed_filelist_file, "a", encoding="utf-8") as fp:
    #     fp.write(preprocessed_fnames+"\n")
    return out_json_obj




app= FastAPI()



loaded_model = load_model() 
print("loaded successfully")

@app.post("/geoLM/test")
async def get_geo_result(req:Request):
    input_json = await req.json()
    image = input_json.get("image")
    img_path = config['PATH']['DATA']
    
    #img_path = '/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/all_words'
    try:
        im = Image.open(BytesIO(base64.b64decode(image)))            
        # image_save = os.path.join(folder_path, "invoice_test")
        if not os.path.exists(img_path):
            os.mkdir(img_path)            
        pic = os.path.join(img_path, 'test_sample.png')
        im.save(pic, 'PNG')
        print("++++++++++==")  
    except:
        flag=1
    # all_words_path = '/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/all_words/Invoice_405_28.json'
    img_path = os.path.join(img_path, 'test_sample.png')


    # with open(all_words_path, 'r') as file:
    #     all_words_ = json.load(file)

    all_words_ = gv_data(img_path)

    pre_data1 = main(all_words_)
    print(pre_data1)




    final_results = []
    for data_ in pre_data1:
        print('>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>>>>>>>')
        # print(data_['form'])
        out_json_obj = preprocessData(data_['form'], img_path)
        print(out_json_obj)
        image_path = out_json_obj['meta']['image_path']
        image = Image.open(image_path)
        pr_labels, geo_results = predict(loaded_model, image, out_json_obj, backbone_type='geolayoutlm')
        print('>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>>>>>')
        print()
        final_results.extend(geo_results[0])
    print(final_results)
    geo_final_result = result_generation(img_path, final_results)
    #result_generation('/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/val_inference_files/dataset/custom_trial__/vis/Invoice_405_28_s_1_linking.png', '/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/val_inference_files/dataset/results/_tagging.json')
    return JSONResponse(content=geo_final_result,status_code=200)


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8165)