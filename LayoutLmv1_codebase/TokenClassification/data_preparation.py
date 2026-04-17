"""
* *********************************************************************************
* Number Theory S/W Pvt. Ltd CONFIDENTIAL                                      *
* *
* [2016] - [2023] Number Theory S/W Pvt. Ltd Incorporated                       *
* All Rights Reserved.                                                          *
* *
* NOTICE:  All information contained herein is, and remains                     *
* the property of Number Theory S/W Pvt. Ltd Incorporated and its suppliers,    *
* if any.  The intellectual and technical concepts contained                    *
* herein are proprietary to Number Theory S/W Pvt. Ltd Incorporated             *
* and its suppliers and may be covered by India. and Foreign Patents,           *
* patents in process, and are protected by trade secret or copyright law.       *
* Dissemination of this information or reproduction of this material            *
* is strictly forbidden unless prior written permission is obtained             *
* from Number Theory S/W Pvt. Ltd Incorporated.                                 *
* *
* *********************************************************************************
"""



"""
Project Name: Trade Finance
Module Name: Lmv2 - Modeling
Flow of execution: First
Dated: Nov 6th, 2023
"""


"""
Trainings:

Date| DocumentName | Author
Nov 11 | Performa Invoice | Tarun Sharma | 13:03 

"""

import os
import cv2
import shutil
import json
import pandas as pd
from functools import cmp_to_key
from google.cloud import vision
from base64 import b64encode
import pytesseract
from configparser import ConfigParser
import os
from typing import List
import cv2
import json
import shutil
from functools import cmp_to_key
import random
import pickle
import psutil
from datetime import datetime
import ast
import tqdm
import utility as tu
from configparser import ConfigParser
from utility import set_basic_config_for_logging, get_logger_object_and_setting_the_loglevel
import time
import random
from tqdm import tqdm
random.seed(12345)

product_code_map : dict = {
"lc" : "LetterOfCredit",
"bg" : "BankGuarantees",
"tp" : "TransportDocument"
}


document_code_map : dict = {
"po": "PurchaseOrder",
"pi": "PerformaInvoice",
"bgc": "BGCancellation",
"lca": "LCApplication",
"bol" : "BillOfLadding",
"awb" : "AirWayBill",
"coo" : "CertificateOfOrigin",
"pl" : "PackingList",
"ic" : "InsuranceCertificate",
"ci" : "CommercialInvoice",
"boe": "BillOfExchange",
"cs" : "CoveringSchedule"
}

zoom = 300/72

  
def contour_sort(a, b):
    return a['x1'] - b['x1'] if abs(a['y1'] - b['y1']) <= 15 else a['y1'] - b['y1']


def remove_garbage(dataset):
    to_remove = ["\u00da", "\u00c6", "\u00c4", "\u00b4", "\u00c5", "Á","\n","|"]
    for key in dataset.keys():
        values = dataset[key]
        for value in values:
            string = value[0]
            new_string = ""
            for char in string:
                if char not in to_remove:
                    new_string+= char
            new_string = new_string.strip()
            value[0] = new_string        


def get_ocr_vision_api(image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/gv_key.json"

    with open(image_path, 'rb') as f:
        ctxt = b64encode(f.read()).decode()

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content = ctxt)

    response = client.text_detection(image=image)

    word_coordinates = []
    all_text = ""

    for i,text in enumerate(response.text_annotations):
        if i != 0:
            vertices = vertices = text.bounding_poly.vertices
            x1 = min(vertex.x for vertex in vertices)
            y1 = min(vertex.y for vertex in vertices)
            x2 = max(vertex.x for vertex in vertices)
            y2 = max(vertex.y for vertex in vertices)
            if x2 - x1 == 0:
                x2 += 1
            if y2 - y1 == 0:
                y2 += 1
            word_coordinates.append({
                "word": text.description,
                "left": x1,
                "top": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
                })
        else:
            all_text = text.description
    return word_coordinates, all_text


def get_ocr_tesseract(img_path, labels_list):
    image = cv2.imread(img_path, 0)
    dataset= {}
    for item in labels_list:
        label= item['label']
        x1= item['x1']
        y1= item['y1']
        x2= item['x2']
        y2= item['y2']
        ROI = image[y1:y2,x1:x2]
        labelled_text = pytesseract.image_to_string(ROI, lang='eng',config='--psm 6')
        if label in list(dataset.keys()):
            dataset[label].append([labelled_text, [x1,y1, x2, y2]])
        else:
            dataset[label] = []
            dataset[label].append([labelled_text, [x1,y1, x2, y2]])
    return dataset



def get_intersection_percentage(bb1, bb2):
    """
    Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
    """
    # print(f"label coords: {bb1}")
    # print(f"ocr coords: {bb2}")
    
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2'] #################################
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    # min_area = min(bb1_area,bb2_area)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
 
    if bb1_area > bb2_area:
        intersection_percent = intersection_area / bb2_area
    else:
        intersection_percent = intersection_area / bb1_area
        if intersection_percent<0.5:
            intersection_percent=1  # if ocr bounding box is big then we need to consider the entire token if the intersection is less than o.5 also 
            
    assert intersection_percent >= 0.0
    assert intersection_percent <= 1.0
    
    # print(f"Intersection Percentage: {intersection_percent}")
    return intersection_percent

def train_test_split(split_file,flag, img_files, labelled_files):
    print("=====================Entered into train and test split code=======================")
    annotation_data: list = []
    logger.info("is annotation_data is instance of list? %s", 
                isinstance(annotation_data, list))
    
    for file in split_file:
        base_name, extension = os.path.splitext(file)
        if extension == ".txt":
            file = file.split(".txt")[0]
        elif extension == ".png":
            file = file.split(".png")[0]
        # exit("+++++++++++++")
        if file  in (img_files and labelled_files):
            if os.path.exists(os.path.join(ocr_path, file + text_file_nme)):
                with open(os.path.join(ocr_path, file + text_file_nme), "r") as f:
                    pdf_text = json.load(f)['word_coordinates']
                f.close()
                for t in pdf_text:
                    t["label"] = "O"
            if os.path.exists(os.path.join(images_path, file + ".png")):
                image = cv2.imread(os.path.join(images_path, file + ".png"))
                h, w, _ = image.shape
                with open(os.path.join(labels_path, file + ".txt"), "r") as f:
                    label = (f.read())
                label = label.split("\n")
                labelled_data = []
                logger.info("is labelled_data is instance of list? %s", isinstance(labelled_data, list))
                f.close()
                if is_agumentation:
                    for l in label:
                        l = l.split()
                        if (len(l) > int(configur['PARAMS'][
                                            'nill_val'])):
                            l_class = dict_mapping[int(l[0])]
                            labelled_data.append({
                                "label": l_class,
                                "x1": int(l[1]),
                                "y1": int(l[2]),
                                "x2": int(l[3]),
                                "y2": int(l[4])
                            })
                else:
                    for l in label:
                        l = l.split()
                        if (len(l) > int(configur['PARAMS']['nill_val'])): #and (int(l[0]) != int(configur['PARAMS']['label_length'])):
                            # if int(l[0]) != int(configur['PARAMS']['label_length']):
                            l_class = dict_mapping[int(l[0])]
                            x_center = float(l[1]) * w
                            y_center = float(l[2]) * h
                            width = float(l[3]) * w
                            height = int(float(l[4]) * h)
                            d_value = int(configur['PARAMS']['div_value'])
                            x0 = int(x_center - (width / d_value))
                            x1 = int(x_center + (width / d_value))
                            y0 = int(y_center - (height / d_value))
                            y1 = int(y_center + (height / d_value))
                            # cv2.rectangle(image, (x0, y0), (x1, y1), (0,255,0), 4)
                            labelled_data.append({
                                "label": l_class,
                                "x1": x0,
                                "y1": y0,
                                "x2": x1,
                                "y2": y1
                            })

                if len(labelled_data) > int(configur['PARAMS']['nill_val']):
                    for data in labelled_data:
                        # print(data)
                        for t in pdf_text:
                            try:
                                intersection_2 = get_intersection_percentage(data, t)
                                if intersection_2 >= float(configur['PARAMS']['percent_val']):
                                    t['label'] = data['label']
                                    #print(data['label'])
                                    # if data['label']=='nostro_bank_name':
                                    #     exit('*****************exited')
                            except:
                                pass

                    for t in pdf_text:
                        if t['label'] == "O":
                            blue_col = int(configur['PARAMS']['color_val'])
                            thick_val = int(configur['PARAMS']['thick_val2'])
                            nul_val = int(configur['PARAMS']['nill_val'])
                            cv2.rectangle(image, (int(t['x1']), int(t['y1'])), (int(t['x2']), int(t['y2'])),
                                        (blue_col, nul_val, nul_val), thick_val)
                        else:
                            if 'S-' in t["label"]:
                                exit('>>>>>>>>>>>>>>> S-  >>>>>>>>>>>>>>')
                            t["label"] = "S-" + t["label"]

                    if len(pdf_text) <= thresh:
                        shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(image_data_path, file + ".png"))
                        final_labelled_data = {
                            "filename": file + ".png",
                            "words": [],
                            "bbox": [],
                            "labels": []
                        }
                        logger.info("is final_labelled_data is instance of dict? %s", isinstance(final_labelled_data, dict))
                        for l_d in pdf_text:
                            final_labelled_data["words"].append(l_d['word'])
                            final_labelled_data["bbox"].append(
                                tu.normalize([l_d['x1'], l_d['y1'], l_d['x2'], l_d['y2']], w, h))
                            final_labelled_data["labels"].append(l_d['label'])
                        annotation_data.append(final_labelled_data)
                        df = pd.DataFrame(final_labelled_data)
                        df.to_csv(f'{SEGREGATION_path}/{file}.csv')                       
                    else:
                        concat_final_label = []
                        final_labelled_data = {
                            "filename": file + ".png",
                            "words": [],
                            "bbox": [],
                            "labels": []
                        }
                        for i, l_d in enumerate(pdf_text):
                            if (i + 1) % thresh == int(configur['PARAMS']['nill_val']):
                                final_labelled_data["filename"] = file + "_s_" + str(int((i + 1) / thresh)) + ".png"
                                shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(image_data_path,
                                                                                                file + "_s_" + str(int((
                                                                                                                                i + 1) / thresh)) + ".png"))
                                annotation_data.append(final_labelled_data)
                                concat_final_label.append(pd.DataFrame(final_labelled_data))
                                final_labelled_data = {
                                    "filename": file + "_s_" + str(int(len(pdf_text) / thresh) + 1) + ".png",
                                    "words": [],
                                    "bbox": [],
                                    "labels": []
                                }
                            final_labelled_data["words"].append(l_d['word'])
                            final_labelled_data["bbox"].append(
                                tu.normalize([l_d['x1'], l_d['y1'], l_d['x2'], l_d['y2']], w, h))
                            final_labelled_data["labels"].append(l_d['label'])
                            #print(final_labelled_data["labels"])
                        if len(final_labelled_data['words']) > int(configur['PARAMS']['nill_val']):
                            shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(image_data_path,
                                                                                            file + "_s_" + str(int(len(
                                                                                                pdf_text) / thresh) + 1) + ".png"))
                            annotation_data.append(final_labelled_data)
                            concat_final_label.append(pd.DataFrame(final_labelled_data))
                        result = pd.concat(concat_final_label, axis=1)
                        result.to_csv(f'{SEGREGATION_path}/{file}.csv')   
                    #print(final_labelled_data["labels"])
            else:
                try:
                    os.remove(os.path.join(labels_path, file + ".txt"))
                except Exception:
                    pass
                try:
                    os.remove(os.path.join(images_path, file + ".png"))
                except Exception:
                    pass
        seed_val = int(configur['PARAMS']['random_seed_val'])
    ratio_val = float(configur['PARAMS']['train_div_ratio'])
    random.seed(seed_val)
    # for i in range (len(annotation_data)):
    #     print(annotation_data[i]['labels'])
    
    #exit()
    if flag=='train':
        random.shuffle(annotation_data)
        train_samples = annotation_data#[:-int(ratio_val * len(annotation_data))]
        len(train_samples)
        train_samples = sorted(train_samples, key=lambda x: x['filename'])
        image_train,words_train, boxes_train, labels_train = tu.data_convert(train_samples)
        if not os.path.exists(os.path.join(folder_path, 'train')):
            os.mkdir(os.path.join(folder_path, 'train'))
        with open(os.path.join(folder_path, 'train.pkl'), 'wb') as t:
            pickle.dump([image_train,words_train, labels_train, boxes_train], t)
        t.close()    
        for t in train_samples:
            shutil.copy(os.path.join(image_data_path, t['filename']), 
                        os.path.join(folder_path, "train", t['filename']))
            
    if flag=='test':
        random.shuffle(annotation_data)
        test_samples = annotation_data#[-int(ratio_val * len(annotation_data)):]
        len(test_samples)
        test_samples = sorted(test_samples, key=lambda x: x['filename'])
        image_test,words_test, boxes_test, labels_test = tu.data_convert(test_samples)
        if not os.path.exists(os.path.join(folder_path, 'test')):
            os.mkdir(os.path.join(folder_path, 'test'))
        with open(os.path.join(folder_path, 'test.pkl'), 'wb') as t:
            pickle.dump([image_test,words_test, labels_test, boxes_test], t)
        t.close()
        for t in test_samples:
            shutil.copy(os.path.join(image_data_path, t['filename']), 
                        os.path.join(folder_path, "test", t['filename']))
                

if __name__== "__main__":
    
    # Setting configuration for logging purposes   
    #####################################################################
    set_basic_config_for_logging(full_folder_path = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/logs", 
                                 filename = f'''data_preparation_{"_".join(str(datetime.now()).split(" "))}''')
    logger = get_logger_object_and_setting_the_loglevel()
        
    process_memory = psutil.Process()
    start_time = datetime.now()
    cpu_utilization_start = psutil.cpu_percent()
    before_memory = process_memory.memory_info().rss
    logger.info(f"checkpoint 2 => setting basis cofiguration for logging purposes")
    # # exit("+++++++++++++++++++")
    # #####################################################################  
    
    
    # ##################################################################
    # print("==================Data Preparation Code===================")
    # product config
    product_config = ConfigParser()
    """
    [OmniParameters]
            products = bills,lc,bg
            document_code = (ci, ic, coo,cs , pl, bol, awb, boe)||(po, pi)||(bgc)
    [Product]
            code = tp
            document_code = [cs]
    """
    # relative path => passed in validation
    # product_config.read("/home/ntlpt39/work/TradeFinance/MVP/hridesh_mvp/training/lmv2-code/src/main/extraction/config/config.ini")
    prod_code = "tp"
    doc_code_ = "awb"
    # if '[' in doc_code:
    #     doc_elements = doc_code[1:-1].split(', ')
    #     # Convert elements to a Python list
    #     doc_code_list = [element.strip() for element in doc_elements]
    # print(doc_code)
    #best_keys_list = ast.literal_eval(configur[f'{ground_truth}_BEST_KEYS']['keys'])
    # data folder path
    # product_wise_folder = ConfigParser()
    # product_wise_folder.read("/home/ntlpt39/work/TradeFinance/MVP/hridesh_mvp/training/lmv2-code/src/main/extraction/config/prod.ini")
    """
    [LetterOfCredit]
        PerformaInvoice = /New_Volume/trade-finance/final_delivery/pi/data/V2/performa_invoice
        PurchaseOrder = /New_Volume/trade-finance/final_delivery/po/data/V2/purchase_order
        LCApplication = ""

    [Bills]
        CoveringSchedule = /New_Volume/trade-finance/final_delivery/pi/data/V2/performa_invoice

    [BankGuarantees]
        BGCancellation = /media/tarun/D3/TradeFinance/final_delivery/lc_cancellation_test

    [TransportDocument]
        CoveringSchedule = /home/ntlpt39/work/TradeFinance/MVP/hridesh_mvp/training/lmv2-code/CS_Test_New

    """


    print(doc_code_)
    doc_code = document_code_map[doc_code_]
    folder_path = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/AWB"
    
    print("==================Trade Finance Solutions===================")  
    
    print(f"Product Code: {prod_code}")
    print(f"Documenry Code: {doc_code}")
    print("folder_path: {}".format(folder_path)) 
    print(f"checkpoint 1 => right product and its path")
    #####################################################################
    # folder_path = '/New_Volume/Rakesh/DATA_LMV3'
    #####################################################################
    configur = ConfigParser()
    configur.read('train_valid.ini')
    # is_agumentation: bool = configur["AGUMENTATION"]["is_agumentation"]  #need to change to 250    
    is_agumentation = False
    
    images_path = os.path.join(folder_path, "Images")
    labels_path = os.path.join(folder_path, "Labels")
    master_path = os.path.join(folder_path, "Master_Data")
    master_labels_path= os.path.join(folder_path, 'Master_Labels')
    ocr_path = os.path.join(folder_path, "OCR")
    iou_path = os.path.join(folder_path, 'Iou_check')
    SEGREGATION_path = os.path.join(folder_path, 'Segregation_Check')
    
    if not os.path.exists(master_path):
        os.makedirs(master_path)
    if not os.path.exists(ocr_path):
        os.makedirs(ocr_path)
    if not os.path.exists(master_labels_path):
        os.makedirs(master_labels_path)
    if not os.path.exists(iou_path):
        os.makedirs(iou_path)
    if not os.path.exists(SEGREGATION_path):
        os.makedirs(SEGREGATION_path)


    with open(os.path.join(folder_path, "label.txt"), "r") as f:
        classes = (f.read())
        classes = classes.split("\n")
    
    print(f"classes in this document \n: {classes}")
    print("Checkpoint 2 =>  right folder creation whatever required in data preparation")
    # exit("++++++++++++++++++++++++++++++++++++++++")
    ####################################################################
    
    
    ####################################################################
    labelled_files: list = os.listdir(labels_path)
    
    # list containing only names of the file
    labelled_files: list = [x.split(".txt")[0] for x in labelled_files]
    
    # list of images
    img_files:list = os.listdir(images_path)
    
    # list containing only the names of the images
    img_files = [x.split(".png")[0] for x in img_files]
    annotation_data: list = []
    list_of_images_having_error_in_reading: list = []
    list_images_having_no_text: list = []
    ######################
    properly_readable_image_files: list = []
    height_list: list = []
    width_list: list = []
    ######################

    thresh: int = int(configur["PARAMS"]["thresh_value"])  #need to change to 250    

    # assert thresh == 300
    ####################################################################
    
    # important variable initialisation of the variables before the loop runs 
    count: int = 1
    assert count == 1
    print("important variables initialisation done before the loop runs")
    print(f"Number of images: {len(img_files)}")
    print("checkpoint 3 => important variables initialisation done")
    # exit("+++++++++++++++++++++")
    
    #################################################################### 
    for file in tqdm(img_files):
        annotate_data: list = []
        word_cordinate_data: list = []
        IOU_value:list = []

        if os.path.exists(os.path.join(labels_path, file + ".txt")):
            
            try:
                image = cv2.imread(os.path.join(images_path, file + ".png"))
                properly_readable_image_files.append(f"{file}/.png")
            except Exception as e:
                list_of_images_having_error_in_reading.append((f"{file}.png", 
                                                            f"{images_path}/{file}.png"))
                print(f"Error reading in Image file name {file}.png")
                continue
            
            word_coordinates:list = []
            ocr_already_available_flag = False 
            image_path = os.path.join(images_path, file)
            directory, base_name = os.path.split(image_path)
            base_name_without_ext, _ = os.path.splitext(base_name)
            ocr_json_path = os.path.join(ocr_path,base_name_without_ext+".json")
            ####################################################################################
            # if ocr present we will not do
            if os.path.exists(ocr_json_path):
                # it will read
                ocr_already_available_flag  = True
                
                json_ocr = json.load(open(os.path.join(ocr_json_path), "r"))
                all_text, word_coordinates = json_ocr["all_text"],json_ocr["word_coordinates"]
                # all_text = json.load(open(os.path.join(master_path, 
                #                                     file + "_all_text.txt"), "r"))
                
                # print(f"word_coordinates: {word_coordinates}")
                # print(f"all_text: {all_text}")
                # exit("+++++++++++++++++++")
            ####################################################################################
            
            
            # with open(os.path.join(ocr_path, file + "_text.txt"), "w") as f:
            #     json.dump({"word_coordinates": word_coordinates}, f)
            
            ####################################################################################
            if len(word_coordinates) == 0:
                list_images_having_no_text.append(f"{file}.png")
                h, w, _ = image.shape
                # print(f"height : {h}, width : {w}")
                height_list.append(h);width_list.append(w)

            else:
                shutil.copy(os.path.join(images_path, file + ".png"), 
                            os.path.join(master_path, file + ".png"))
                
                shutil.copy(os.path.join(labels_path, file + ".txt"), 
                            os.path.join(master_path, file + "_LabelImg.txt"))
                # dumping ocr text, coordinates in master folder
                with open(os.path.join(master_path, file + "_text.txt"), "w") as f:
                    json.dump({"word_coordinates": word_coordinates}, f)
                if not ocr_already_available_flag:
                    
                    # dumping ocr text, coordinates in ocr folder
                    with open(os.path.join(ocr_path, file + "_text.txt"), "w") as f:
                        json.dump({"word_coordinates": word_coordinates}, f)
                    
                    # dumping all text into master folder
                    try:
                        with open(os.path.join(master_path, file + "_all_text.txt"), "w") as f:
                            json.dump({"all_text": all_text}, f)
                    except Exception as e:
                        print("exception : {e}")
                        pass
                
                ###########################################
                h, w, _ = image.shape
                height_list.append(h);width_list.append(w)
                ###########################################
                
                with open(os.path.join(labels_path, file + ".txt"), "r") as f:
                    label = (f.read())
                label = label.split("\n")
                labelled_data:list = []
                # print("label : {}".format(label))
                # print("word coordinates : {}".format(word_coordinates))
                # exit("+++++++++++++++++++")

                if not is_agumentation:
                    for l in label:
                        l = l.split()
                        if len(l) > 0:
                            l_class = classes[int(l[0])]
                            
                            # centre of the image 
                            ############################
                            x_center = float(l[1]) * w
                            y_center = float(l[2]) * h
                            ############################
                            
                            width = float(l[3]) * w
                            height = int(float(l[4]) * h)
                            
                            x0 = int(x_center - (width / 2))
                            x1 = int(x_center + (width / 2))
                            y0 = int(y_center - (height / 2))
                            y1 = int(y_center + (height / 2))
                            
                            cv2.rectangle(image, (x0, y0), (x1, y1), (0,255,0), 4)
                            
                            labelled_data.append({
                                "label": l_class,
                                "x1": x0,
                                "y1": y0,
                                "x2": x1,
                                "y2": y1
                            })
                        
                if len(labelled_data) > 0:
                    dataset = {}
                    
                    for data in labelled_data:
                        # print(f"data: {data}")
                        overlapping_boxes: list = []
                        labelled_text: str = ""
                        
                        for t in word_coordinates:
                            # t  = eval(t)
                            # print("t value: ", t)
                            # print(f"type t value: {type(t)}")                            
                            # print(f"intersection percentage: {get_intersection_percentage(data, t)}")
                            # exit("++++++++++")
                            try:
                                annotate_data.append(data)
                                word_cordinate_data.append(t)
                                IOU_value.append(get_intersection_percentage(data, t))
                                
                                if get_intersection_percentage(data, t) >= \
                                    float(configur["PARAMS"]["percent_val"]):
                                    
                                    t['label'] = data['label']
                                    overlapping_boxes.append(t)
                            except Exception as e:
                                print("exception : " + str(e))
                        
                        for t in overlapping_boxes:
                            if len(labelled_text) == 0:
                                labelled_text = t['word']
                            else:
                                labelled_text += " " + t['word']
                        
                        # if len(labelled_text.strip()) == 0:
                        #     print()
                            # print(file + " - " + str(data) + " - " + str(len(overlapping_boxes)))
                        else:
                            if data['label'] in list(dataset.keys()):
                                dataset[data['label']].append([labelled_text, 
                                                            [data['x1'], data['y1'], 
                                                                data['x2'], data['y2']]])
                            else:
                                dataset[data['label']] = []
                                dataset[data['label']].append([labelled_text, 
                                                            [data['x1'], data['y1'], 
                                                                data['x2'], data['y2']]])
                    if dataset == {}:
                        print(file + " - blank")
                        
                    remove_garbage(dataset)
                    
                    with open(os.path.join(master_path, file + "_labels.txt"), "w") as f:
                        json.dump(dataset, f)
                    with open(os.path.join(master_labels_path, file + "_labels.txt"), "w") as f:
                        json.dump(dataset, f)
                    
                    csv_data = {'annotate_data':annotate_data, 
                                'word_cordinate_data':word_cordinate_data,
                                'IOU_value':IOU_value}
                    
            try:
                df = pd.DataFrame(csv_data)
                df.to_csv(f'{iou_path}/{file}.csv', index = False)
            except:
                pass
        # else:
        #     # print("No") 
        #     print()
        
    # exit("+++++++++++++++")
    
    image_data = pd.DataFrame({"image_name": [f"{img}.png" for img in properly_readable_image_files],
                                            "height": height_list,
                                            "width": width_list})
                    
    problematic_images = pd.DataFrame({"image_name": [file[0] for file in list_of_images_having_error_in_reading], 
                                                    "path": [file[1] for file in list_of_images_having_error_in_reading]})
                    
    image_having_no_text = pd.DataFrame({"image_name": list_images_having_no_text})


    image_data.to_csv(f'{master_path}/image_information.csv', index=False)
    problematic_images.to_csv(f'{master_path}/probelmatic_images.csv', index=False)
    image_having_no_text.to_csv(f"images_having_no_text.csv", index=False)

    with open(os.path.join(folder_path, "label.txt"), "r") as file:
        class_names: List = file.readlines()
        class_names = list(map(lambda x: x.strip(), class_names))
        logger.info("is class_names is a instance of list? %s", isinstance(class_names, list))
        dict_mapping = dict(enumerate(class_names))
        logger.info("is dict_mapping is a instance of dict? %s", isinstance(dict_mapping, dict))

    file.close()

    labelled_files = os.listdir(labels_path)
    labelled_files = [x.split(".txt")[0] for x in labelled_files]
    img_files = os.listdir(images_path)
    img_files = [x.split(".png")[0] for x in img_files]

    logger.info("is labelled_files is a instance of list? %s", isinstance(labelled_files, list))
    # using intersection percentage
    annotation_data = []
    logger.info("is annotation_data is instance of list? %s", isinstance(annotation_data, list))

    thresh = int(configur['PARAMS']['thresh_value'])  # 300

    logger.info("is threshold value is instance of int? %s", isinstance(thresh, int))
    text_file_nme = ".json"

    zoom = int(configur['PARAMS']['thresh_value']) / int(configur['PARAMS']['zoom_val'])  # 300 / 72

    image_data_path = os.path.join(folder_path, "Images_Data")

    if not os.path.exists(ocr_path):
        os.mkdir(ocr_path)
    if not os.path.exists(image_data_path):
        os.mkdir(image_data_path)

    thresh = int(configur['PARAMS']['thresh_value'])  # 300
    seed_val = int(configur['PARAMS']['random_seed_val'])
    ratio_val = float(configur['PARAMS']['train_div_ratio'])
    random.seed(seed_val)
    random.shuffle(labelled_files)

    #Uncomment if you do not have train.txt file and test.txt file
    
    # train_samples_imgs = labelled_files[:-int(ratio_val * len(labelled_files))]
    # print(len(train_samples_imgs))
    # test_samples_imgs = labelled_files[-int(ratio_val * len(labelled_files)):]
    # print(len(test_samples_imgs))
    print(len(img_files)) 
    os.path.join(folder_path, "label.txt")
    try:
        with open(os.path.join(folder_path, "train.txt"), "r") as f:
            train_samples_imgs = [line.strip() for line in f.readlines()]
            train_samples_imgs = set(train_samples_imgs)
        print("length of train samples: ", len(train_samples_imgs))

        with open(os.path.join(folder_path, "test.txt"), "r") as f:
            test_samples_imgs = [line.strip() for line in f.readlines()]
            test_samples_imgs = set(test_samples_imgs)
        print("length of test samples: ", len(test_samples_imgs))
    except:
        print("WARINING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Split not provided. Doing random split.")
        random.shuffle(img_files)
        random.shuffle(img_files)
        split_index = int(0.8 * len(img_files)) 
        train_samples_imgs = img_files[:split_index]
        test_samples_imgs = img_files[split_index:]
        with open('train.txt', 'w') as train_file:
            train_file.write('\n'.join(train_samples_imgs))

        with open('test.txt', 'w') as test_file:
            test_file.write('\n'.join(test_samples_imgs))


    # exit("+++++++++++")
    
    #####################################################################
    train_test_split(train_samples_imgs,'train', img_files, labelled_files)
    train_test_split(test_samples_imgs,'test', img_files, labelled_files)


    end_time = datetime.now()
    cpu_utilization_end = psutil.cpu_percent()
    diff = end_time - start_time
    after_memory = process_memory.memory_info().rss
    cpu_utt = cpu_utilization_end - cpu_utilization_start
    memory_consumption = after_memory - before_memory
    logger.info("total time taken for data preparation:" + str(diff))
    logger.info("cpu_utilization %:" + str(cpu_utt))
    logger.info("memory_consumption in bytes:" + str(memory_consumption))
    logger.info(('RAM memory % used:', psutil.virtual_memory()[2]))




