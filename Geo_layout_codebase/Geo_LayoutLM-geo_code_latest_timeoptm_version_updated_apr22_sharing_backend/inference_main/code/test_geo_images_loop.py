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
import torch.nn.functional as F
from glob import glob
import imagesize
from transformers import BertTokenizer
from typing import Dict
from prediction_utility import result_generation
from result_utility import get_eval_kwargs_geolayoutlm_vie, getitem_geo
# from prediction_utility import result_generation
from pre_process_utility import main, gv_data
from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request,Depends, Header
# from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi import FastAPI,Request
from io import BytesIO
import base64
from model_saving_eval_mode import save_loaded_model
import os
from google.cloud import vision
import pytesseract
from tqdm import tqdm
import time
gv_key = '/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/gv_key.json'



App_Filepath = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config.read(App_Filepath + '/config.ini')

# geo_clsses_path = config['PATH']['GEO_CLASSES_PATH']
# MODEL_PATH = config['PATH']['MODEL_PATH']

geo_dump_dir = config['PATH']['GEO_DUMP']

document_model_path = {
    "AWB": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/awb_epoch=34-f1_labeling=0.9682.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/AWB/class_names.txt"
    },
    "BOE": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/boe_epoch=12-f1_labeling=0.8897.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/BOE/class_names.txt"
    },
    "BOL": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/bol_epoch=39-f1_labeling=0.9252.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/BOL/class_names.txt"
    },
    "CI": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/ci_epoch=20-f1_labeling=0.9313.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/CI/class_names.txt"
    },
    "COO": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/coo_epoch=27-f1_labeling=0.9249.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/COO/class_names.txt"
    },
    "CS": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/cs_epoch=3-f1_labeling=0.8830.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/CS/class_names.txt"
    },
    "IC": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/ic_epoch=42-f1_labeling=0.8804.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/IC/class_names.txt"
    },
    "PL": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/pl_epoch=11-f1_labeling=0.9244.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/PL/class_names.txt"
    },
    "PI": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/pi_epoch=22-f1_labeling=0.9285.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/PI/class_names.txt"
    },
    "PO": {
        "model_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/geo_models_eval_mode_save/po_epoch=9-f1_labeling=0.9165.pth",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/PO/class_names.txt"
    },
    "LC_Can": {
        "model_path": "../Trained_Models/Extraction/LCCancellation/Best_Model",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/LCCancellation/class_names.txt"
    },
    "CERTIFICATE": {
        "model_path": "../Trained_Models/Extraction/CERTIFICATE/Best_Model",
        "class_file_path": "/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/BEST_KEYS/CERTIFICATE/class_names.txt"
    }
}


def get_ocr_vision_api_charConfi(image_path):
	"""
    Performs OCR (Optical Character Recognition) using Google Cloud Vision API and returns character-level confidence scores.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing formatted results (list of dictionaries) and all the extracted text (str).
    """
	# Initialize the Vision API client
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gv_key 
	client = vision.ImageAnnotatorClient()

	# Load the image
	try:
		image_file = open(image_path, 'rb')
		image_data = image_file.read()
	except Exception as e:
		raise HTTPException(status_code=400, detail={"message": "{}".format(e), "error_code": "501"})
	finally:
		if hasattr(image_file,"close"):
			image_file.close()

	# Perform text detection
	image = vision.Image(content=image_data)
	response = client.document_text_detection(image=image)
	#print(dir(response))
	# Initialize a list to store the formatted results
	formatted_results = []

	# Initialize a string to store all the extracted text
	all_extracted_text = ""

	# Extract and format the text and bounding box information
	for page in response.full_text_annotation.pages:
		for block in page.blocks:
			for paragraph in block.paragraphs:
				for word in paragraph.words:
					word_text = "".join([symbol.text for symbol in word.symbols])
					confidence = word.confidence
					_ = [symbol.confidence for symbol in word.symbols]
					vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
					x1 = min([v[0] for v in vertices])
					x2 = max([v[0] for v in vertices])
					y1 = min([v[1] for v in vertices])
					y2 = max([v[1] for v in vertices])
					formatted_word = {
						"word": word_text,
						"left": x1,
						"top": y1,
						"width": x2 - x1,
						"height": y2 - y1,
						# "confidence": confidence,
						# 'char_confi': char_confidences,  # List of character-level confidences
						"x1": x1,
						"y1": y1,
						"x2": x2,
						"y2": y2,
					}
					formatted_results.append(formatted_word)
					all_extracted_text += word_text + ' '
	print(all_extracted_text)
	return formatted_results, all_extracted_text


def load_model_weight(net, device_m, MODEL_PATH_IN): 
    print("Loading ckpt from:", MODEL_PATH_IN)
    pretrained_model_state_dict = torch.load(MODEL_PATH_IN, map_location=device_m)
    print("HERE")
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




def load_model(geo_clsses_path, MODEL_PATH_IN):
    mode = "val"
    geo_cfg = get_config(geo_clsses_path, './configs/finetune_funsd.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # geo_cfg.dump_dir = '/New_Volume/number_theory/GEO_Rakesh/master_table_extraction/data/output/invoice_test/headers/geo_out'
    
    # pt_list = os.listdir(os.path.join(geo_cfg.workspace, "checkpoints"))
    # if len(pt_list) == 0:
    #     print("Checkpoint file is NOT FOUND!")
    # pt_to_be_loaded = pt_list[0]
    # if len(pt_list) > 1:
    #     # import ipdb;ipdb.set_trace()
    #     for pt in pt_list:
    #         if geo_cfg[mode].pretrained_best_type in pt:
    #             pt_to_be_loaded = pt
    #             break
            
    # geo_cfg.pretrained_model_file = os.path.join(geo_cfg.workspace, "checkpoints", pt_to_be_loaded)
    # print(geo_cfg)
    geo_cfg.pretrained_model_file = ''
    net = get_model(geo_cfg)
    load_model_weight(net, device, MODEL_PATH_IN)#, geo_cfg.pretrained_model_file)
    # net.to("cpu")
    # net.eval()
    
    # Move model to the correct device (CUDA or CPU)
    net.to(device)
    # Set model to evaluation mode
    net.eval()
    # save_loaded_model(net, '/media/ntlpt19/5250315B5031474F/TradeFinance_Geo/eval_mode_save', filename = pt_filename)
    # exit('>>>>>>>>.')
    
    # If you need the device the model is on, you can still check it
    # current_device = next(net.parameters()).device
    # print(f"Model is on device: {current_device}")
    
    
    return net


def predict(net, image, json_obj,eval_kwargs, backbone_type='geolayoutlm'):
    tokenizer = net.tokenizer
    input_data = getitem_geo(image, json_obj, tokenizer, backbone_type)
    print('input_data >>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print(input_data)
    # net.to("cuda")
    # net.eval()
    #device = 'cpu'
    # model.to(device)
    ####################################################
    # device = next(net.parameters()).device
    ####################################################
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model to the correct device (CUDA or CPU)
    net.to(device)
    # Set model to evaluation mode
    net.eval()
    current_device = next(net.parameters()).device
    print(f"Model is on device: {current_device}")
    
    # Ensure CUDA is available
    print(f"Model is on device: {device}")
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







def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read the binary data of the image
        image_binary = image_file.read()

        # Encode the binary data into Base64
        base64_encoded = base64.b64encode(image_binary)

        # Decode the bytes to a UTF-8 string
        base64_string = base64_encoded.decode("utf-8")

    return base64_string







def get_geo_result_final(image_base64, file_path, OCR_path, loaded_model, eval_kwargs):
    # input_json = await req.json()
    # image = input_json.get("image")
    # image = image_base64
    # img_path = config['PATH']['DATA']
    
    #img_path = '/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/all_words'
    # try:
    #     im = Image.open(BytesIO(base64.b64decode(image)))            
    #     # image_save = os.path.join(folder_path, "invoice_test")
    #     if not os.path.exists(img_path):
    #         os.mkdir(img_path)            
    #     pic = os.path.join(img_path, file_name)
    #     im.save(pic, 'PNG')
    #     print("++++++++++==")  
    # except:
    #     flag=1
    # # all_words_path = '/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/all_words/Invoice_405_28.json'
    # img_path = os.path.join(img_path, file_name)
    file_name = os.path.basename(file_path)
    print(file_name)
    if '.png' in file_name:
        ocr_file = file_name.replace('.png', '_textAndCoordinates.txt')
        # if not os.path.exists(os.path.join(OCR_path,ocr_file)):
        #     ocr_file = file_name.replace('.png', '_text.txt')
        if not os.path.exists(os.path.join(OCR_path,ocr_file)):
            ocr_file = file_name.replace('.png', '_coordinates.txt')
        if not os.path.exists(os.path.join(OCR_path,ocr_file)):
            formatted_results, all_extracted_text = get_ocr_vision_api_charConfi(file_path)
            text_file_path = os.path.join(OCR_path, f"{file_name[:-4]}_text.txt")
            coordinates_file_path = os.path.join(OCR_path, f"{file_name[:-4]}_coordinates.txt")
            # Save extracted text
            with open(text_file_path, 'w') as text_file:
                text_file.write(all_extracted_text)
            # Save coordinates
            with open(coordinates_file_path, 'w') as coord_file:
                coord_file.write(str({"word_coordinates": formatted_results}))
            ocr_file = coordinates_file_path
            
    if '.jpg' in file_name:
        ocr_file = file_name.replace('.jpg', '_textAndCoordinates.txt')
        # if not os.path.exists(os.path.join(OCR_path,ocr_file)):
        #     ocr_file = file_name.replace('.jpg', '_text.txt')
        if not os.path.exists(os.path.join(OCR_path,ocr_file)):
            ocr_file = file_name.replace('.jpg', '_coordinates.txt')
        if not os.path.exists(os.path.join(OCR_path,ocr_file)):
            formatted_results, all_extracted_text = get_ocr_vision_api_charConfi(file_path)
            text_file_path = os.path.join(OCR_path, f"{file_name[:-4]}_all_text.txt")
            coordinates_file_path = os.path.join(OCR_path, f"{file_name[:-4]}_coordinates.txt")
            # Save extracted text
            with open(text_file_path, 'w') as text_file:
                text_file.write(all_extracted_text)
            # Save coordinates
            with open(coordinates_file_path, 'w') as coord_file:
                coord_file.write(str({"word_coordinates": formatted_results}))
            ocr_file = coordinates_file_path
            
    # with open(all_words_path, 'r') as file:
    #     all_words_ = json.load(file)
    all_words_ = gv_data(file_path, ocr_file = os.path.join(OCR_path,ocr_file))
    pre_data1 = main(all_words_)

    geo_final_results = []
    for data_ in pre_data1:
        print('>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>>>>>>>')
        # print(data_['form'])
        out_json_obj = preprocessData(data_['form'], file_path)
        image_path = out_json_obj['meta']['image_path']
        image = Image.open(image_path)
        pr_labels, geo_results = predict(loaded_model, image, out_json_obj, eval_kwargs, backbone_type='geolayoutlm')
        # print(geo_results)
        geo_final_results.extend(geo_results[0])
    # exit('OKKKKKKKKKKKKKKKKKKKKKK')
    geo_final_result = result_generation(file_path, geo_final_results, all_words_)
    print(file_name)
    print(geo_final_result)

    
    #result_generation('/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/val_inference_files/dataset/custom_trial__/vis/Invoice_405_28_s_1_linking.png', '/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/val_inference_files/dataset/results/_tagging.json')
    # return JSONResponse(content=geo_final_result,status_code=200)


log_file = "processing_times.txt"

import time
if __name__ == '__main__':
    Master_folder = '/home/ntlpt19/TF_testing_EXT/dummy_responces/itf_observation_new/Samples/Document_Images/19801'
    OCR_path = '/home/ntlpt19/TF_testing_EXT/dummy_responces/itf_observation_new/Samples/OCR'
    os.makedirs(geo_dump_dir, exist_ok=True)
    
    with open(log_file, "a") as f:
        for doctype in os.listdir(Master_folder):
            if doctype in document_model_path:
                loaded_model = load_model(document_model_path[doctype]['class_file_path'], document_model_path[doctype]['model_path']) 
                eval_kwargs = get_eval_kwargs_geolayoutlm_vie(document_model_path[doctype]['class_file_path'])

                print("loaded successfully")
                images_path = os.path.join(Master_folder, doctype)
                for images_files in os.listdir(images_path):
                    # if images_files not in ['IM-000000016466851-AP_page_0.png']:
                    #     continue file_name = os.path.basename(file_path)
                    print((os.path.join(geo_dump_dir, os.path.splitext(images_files)[0] + '.png')))
                    if (not os.path.exists(os.path.join(geo_dump_dir, images_files)) and 
                        not os.path.exists(os.path.join(geo_dump_dir, os.path.splitext(images_files)[0] + '.png'))):
                    # if not os.path.exists(os.path.join(geo_dump_dir, images_files)) and not os.path.exists(os.path.join(geo_dump_dir, os.path.splitext(images_files)[0])+'.jpg'):
                        image_base64_ = image_to_base64(os.path.join(images_path, images_files))
                        start_time = time.time()
                        get_geo_result_final(image_base64_, os.path.join(images_path, images_files), OCR_path, loaded_model, eval_kwargs)
                        time_taken = time.time() - start_time
                        print(f"TIME TAKEN FOR RESULT GENERATION {time_taken:.2f} seconds.")
                        f.write(f"{images_files}, {time_taken:.2f} seconds\n")
                    else:
                        print(f'File {images_files} already exists.')
                        f.write(f"{images_files} already exists.\n")
                        
                        
