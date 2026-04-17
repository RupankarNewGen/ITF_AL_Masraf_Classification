import argparse
import os
import shutil
import time
import csv, re
import random
from pathlib import Path
import pandas as pd
import cv2
import torch
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import traceback
from fuzzywuzzy import fuzz
import ast

#  source /datadrive/khushal/idp39/bin/activate
idp_inv_images_folder = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations/Eval_data/Images"
idp_inv_labels_folder = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations/Eval_data/Labels"
idp_inv_ocr_folder = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations/Eval_data/OCR"
classes_path = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations/Eval_data/class_names.txt"
annot_classses_file = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations/Eval_data/label.txt" 
idp_inv_json_results = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations/Eval_data/final_results"
idp_inv_image_results = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations/Eval_data/reports__nov6"
csv_file_path = './Inv_geo_OUTPUT_nov6__'
plot_gt_flag = False
idp_model_type = "GEOlayoutLMVForTokenClassification"
itf_ocr = True



def save_fuzzy_results(label_wise_total_pred_count, label_wise_fuzz_pred_count, folder_name="geo_reports", filename="fuzzy_results.xlsx"):
    """
    Saves fuzzy matching results to an Excel file in a specified folder.

    Parameters:
    - label_wise_total_pred_count (dict): Dictionary with total count for each label.
    - label_wise_fuzz_pred_count (dict): Dictionary with fuzzy match count for each label.
    - folder_name (str): Folder where the Excel file will be saved.
    - filename (str): Name of the Excel file to save results.
    """
    # Initialize lists to store data for each label
    data = []

    # Iterate over each label in the total prediction count dictionary
    for label, actual_count in label_wise_total_pred_count.items():
        # Get the predicted count from fuzz dictionary, default to 0 if label not found
        predicted_count = label_wise_fuzz_pred_count.get(label, 0)

        # Calculate accuracy, ensuring no division by zero
        accuracy = round((predicted_count / actual_count * 100), 4) if actual_count > 0 else 0

        # Append the row data to the list
        data.append([label, predicted_count, actual_count, accuracy])

    # Convert the data into a DataFrame
    df = pd.DataFrame(data, columns=["label", "predicted count", "actual count", "accuracy"])

    # Ensure the specified folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Define the full path for the output file
    output_path = os.path.join(folder_name, filename)

    # Save the DataFrame to an Excel file
    df.to_excel(output_path, index=False)
    
    print(f"Results successfully saved to {output_path}")

def save_overall_geo_report(total_actual_labels, total_pred_labels, fuzz25_correct_preds, fuzz50_correct_preds, 
                            fuzz75_correct_preds, fuzz85_correct_preds, fuzz90_correct_preds, fuzz100_correct_preds, 
                            precision_pt=4, folder_name="geo_reports", filename="overall_geo_report.xlsx"):
    """
    Saves overall geo report metrics to an Excel file in a specified folder.

    Parameters:
    - total_actual_labels (int): Total count of actual labels.
    - total_pred_labels (int): Total count of predicted labels.
    - fuzz25_correct_preds, fuzz50_correct_preds, fuzz75_correct_preds, fuzz85_correct_preds, fuzz90_correct_preds, fuzz100_correct_preds (int): Correct predictions at different fuzzy thresholds.
    - precision_pt (int): Decimal points for rounding precision and recall (default is 4).
    - folder_name (str): Folder where the Excel file will be saved.
    - filename (str): Name of the Excel file to save results.
    """
    # Calculate precision and recall for each threshold
    metrics_data = [
        ["Total Actual Labels", total_actual_labels],
        ["Total Predictions", total_pred_labels],
        ["Precision (fuzz>=25)", round(fuzz25_correct_preds / total_pred_labels, precision_pt)],
        ["Precision (fuzz>=50)", round(fuzz50_correct_preds / total_pred_labels, precision_pt)],
        ["Precision (fuzz>=75)", round(fuzz75_correct_preds / total_pred_labels, precision_pt)],
        ["Precision (fuzz>=90)", round(fuzz90_correct_preds / total_pred_labels, precision_pt)],
        ["Precision (fuzz>=100)", round(fuzz100_correct_preds / total_pred_labels, precision_pt)],
        ["Recall (fuzz>=25)", round(fuzz25_correct_preds / total_actual_labels, precision_pt)],
        ["Recall (fuzz>=50)", round(fuzz50_correct_preds / total_actual_labels, precision_pt)],
        ["Recall (fuzz>=75)", round(fuzz75_correct_preds / total_actual_labels, precision_pt)],
        ["Recall (fuzz>=85)", round(fuzz85_correct_preds / total_actual_labels, precision_pt)],
        ["Recall (fuzz>=90)", round(fuzz90_correct_preds / total_actual_labels, precision_pt)],
        ["Recall (fuzz>=100)", round(fuzz100_correct_preds / total_actual_labels, precision_pt)]
    ]

    # Convert to DataFrame for easy export to Excel
    df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])

    # Ensure the specified folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Define the full path for the output file
    output_path = os.path.join(folder_name, filename)

    # Save the DataFrame to an Excel file
    df.to_excel(output_path, index=False)

    print(f"Overall report successfully saved to {output_path}")

def read_labels_from_file(file_path):
    """
    Reads a label.txt file and returns a list of labels.

    Parameters:
    - file_path: Path to the label.txt file.

    Returns:
    - A list of labels.
    """
    with open(file_path, 'r') as file:
        # Read lines and strip any trailing whitespace/newline characters
        labels = [line.strip() for line in file.readlines()]
    
    return labels

def read_classes(file_path):
    with open(file_path, 'r') as file:
        # Read the single line and convert the string representation of the list to an actual list
        classes = ast.literal_eval(file.readline().strip())
    return classes

# Function to create label2id and id2label mappings
def create_label_mappings(labels):
    label2id = {}
    id2label = {}
    
    # Enumerate through the labels to create mappings
    for idx, label in enumerate(labels):
        # Directly use the label as it is, without modifying
        label2id[label] = idx
        id2label[idx] = label
    
    return label2id, id2label

# model_path = '/home/khushal/Desktop/data_n_models/Models/invoice_extraction/lmv2_aug_22/layoutLMV2ForTokenClassification_b4_final_best.pth'

# Get current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(idp_inv_image_results, exist_ok=True)

def plot_boxes_on_image(image_path, pred, actual_value, save_path, file, plot_gt = True):
  """
  Plots bounding boxes, class labels, and confidence scores on an image.

  Args:
    image_path: Path to the image file.
    boxes: A list of bounding boxes, where each box is a tuple (x1, y1, x2, y2).
    classes: A list of class labels for each bounding box.
    confidence_scores: A list of confidence scores for each bounding box.
  """
  boxes = [] ; classes = [] ; confidence_scores = []

  for b,l,c,_ in pred:
      boxes.append(b)
      classes.append(l)
      confidence_scores.append(c)
  
  img = cv2.imread(image_path)

  for box, cls, conf in zip(boxes, classes, confidence_scores):
    print(box, cls, conf)
    x1, y1, x2, y2 = [int(b) for b in box]
    label = f"{cls}: {conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  
  if plot_gt:
    img2 = cv2.imread(image_path)
    boxes = [] ; classes = []
    for info in actual_value:
      boxes.append(info["bbox"])
      classes.append(info["class"])

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        label = f"{cls}: 1"
        cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  
    
    os.makedirs(os.path.join(save_path,"original"), exist_ok=True)
    cv2.imwrite(os.path.join(save_path, "original", file.rsplit(".",1)[0]+ "_original."+file.rsplit(".",1)[1]), img2)

  #   cv2.imshow("Image with Boxes", img)
  #   cv2.waitKey(0) ; cv2.destroyAllWindows()

  cv2.imwrite(os.path.join(save_path, file), img)
  


def calculate_iou(bbox1, bbox2):
    """
    Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    iou_percent = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou_percent

def merge_words_in_bbox(ocr_data, bbox):
    merged_words = []
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    for word_info in ocr_data:
        # word_bbox = word_info['coordinates']
        # word_x1, word_y1, word_x2, word_y2 = word_bbox
        word_x1, word_y1, word_x2, word_y2 = word_info["x1"], word_info["y1"], word_info["x2"], word_info["y2"]
        word_center_x = (word_x1 + word_x2) / 2
        word_center_y = (word_y1 + word_y2) / 2
        if bbox_x1 <= word_center_x <= bbox_x2 and bbox_y1 <= word_center_y <= bbox_y2:
            merged_words.append(word_info)
    
    # merged_words = [x['word'] for x in sorted(merged_words, key=lambda x: (int(x['top']/5), x['left']))]
    merged_words = [x['word'] for x in merged_words]
    return ' '.join(merged_words)


def get_actual_value(ocr_data, label_data, img_width, img_height, idx2label):
    actual_value = {}
    for annotation in label_data:
        class_id = annotation['class_id']
        label = idx2label[class_id]
        x_center, y_center = annotation['x_center'] * img_width, annotation['y_center'] * img_height
        width, height = annotation['width'] * img_width, annotation['height'] * img_height
        bbox = [int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(y_center + height / 2)]
        if class_id not in actual_value.keys():
            actual_value[class_id] =[{
                'words': merge_words_in_bbox(ocr_data, bbox),
                'bbox': bbox,
                'class_id': class_id,
                'class': label
            }]
        else:
            actual_value[class_id].append({
                'words': merge_words_in_bbox(ocr_data, bbox),
                'bbox': bbox,
                'class_id': class_id,
                'class': label
            })
        
    return actual_value


def read_ocr_data(ocr_file):
    ocr_file = os.path.join(idp_inv_ocr_folder, ocr_file)
    print("OCR File : ", ocr_file)
    try:
        with open(ocr_file, 'r') as f:
            ocr_data = eval(f.read())
        return ocr_data
    except Exception as e:
        print(f"Error reading OCR file '{ocr_file}': {str(e)}")
        traceback.print_exc()
        return None

def read_xml_label_data(label_file):
    label_file = os.path.join(idp_inv_labels_folder, label_file)
    try:
        label_data = []
        tree = ET.parse(label_file)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Convert bounding box to YOLO format (optional)
            width = float(root.find('size/width').text)
            height = float(root.find('size/height').text)
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # Assuming you have a dictionary to map class names to class IDs
            class_id = class_name_to_id(class_name)

            label_data.append({
                'class_id': class_id,
                'class': class_name,
                'x_center': x_center,
                'y_center': y_center,
                'width': bbox_width,
                'height': bbox_height,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        return label_data
    except Exception as e:
        print(f"Error reading label file '{label_file}': {str(e)}")
        return []

def class_name_to_id(class_name):
    class_map = {'table': 0}
    return class_map.get(class_name) #, -1)  # Return -1 if the class is not found


def read_text_label_data(label_file):
    label_file = os.path.join(idp_inv_labels_folder, label_file)
    try:
        label_data = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                label_data.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
        return label_data
    except Exception as e:
        print(f"Error reading label file '{label_file}': {str(e)}")
        return []

# Function to preprocess address fields by removing special characters
def preprocess_address(address):
    return re.sub(r'[^\w\s]', '', address).lower()

# Function to standardize and compare date fields
def standardize_date(date_str):
    for fmt in ('%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt).strftime('%d-%m-%Y')
        except ValueError:
            pass
    return None

def calculate_fuzzy_score(value1, value2, field_name):
    if pd.isna(value1) or pd.isna(value2):
        return 0

    # Standardize the values for comparison
    value1 = str(value1).strip().lower()
    value2 = str(value2).strip().lower()
    # Remove [SEP] or SEP only if it's at the end
    if value2.endswith("[sep]"):          #[SEP]
        value2 = str(value2.rstrip("[sep]").strip())
    elif value2.endswith("sep"):
        value2 = str(value2.rstrip("sep").strip())
    else:
        pass
    value1 = value1.replace(" ", "")
    value2 = value2.replace(" ", "")

    if 'date' in field_name.lower():
        standardized_date1 = standardize_date(value1)
        standardized_date2 = standardize_date(value2)
        print("!!!!!!", standardized_date1, standardized_date2)
        return 100 if value1 == value2 else fuzz.ratio(value1, value2)
        # return 100 if standardized_date1 == standardized_date2 else fuzz.ratio(standardized_date1, standardized_date2)
        # return 100 if standardized_date1 == standardized_date2 else fuzz.partial_ratio(standardized_date1, standardized_date2)
    elif 'address' in field_name.lower():
        value1 = preprocess_address(value1)
        value2 = preprocess_address(value2)
        return fuzz.ratio(value1, value2) # fuzz.partial_ratio(value1, value2)
    else:
        return fuzz.ratio(value1, value2) # fuzz.partial_ratio(value1, value2) 

def detect(save_csv=False):
    # imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, view_img, save_txt = opt.output, opt.view_img, opt.save_txt
    # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = opt.device
    if os.path.exists(out):
        print("Output path exists !!!")
        # exit()
        # out = out + str(datetime.now())
        # shutil.rmtree(out)  # delete output folder
    os.makedirs(out, exist_ok=True)  # make new output folder

    # # Get names and colors
    names = ["table"]# load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # CSV File for Results
    csv_path = os.path.join(out, './lmv2-detection-results.csv')
    # with open(idp_inv_classes_file,"r") as f:
    #     classes = [x.strip() for x in f.read().split("\n") if x != ""]
    # id2label = {i:x for i,x in enumerate(classes)}
    # label2id = {x:i for i,x in enumerate(classes)}
    labels = read_labels_from_file(classes_path)
    label2id, id2label = create_label_mappings(labels)
    classes_info= read_labels_from_file(annot_classses_file)
    print(f"annoatation labels: {classes_info}")
    # classes_info = ["invoice_number","invoice_date","bill_to","ship_to","remit_to","customer_no","invoice_total","sales_order_number","order_number","customer_order_number","order_date","due_date","page_number","ship_date","account_number","swift_code","purchase_order_number","amount","cash_discount","charge_type","doc_curr","im_vat_no","net_amount","tax_amount","tax_percent","vat_code","vendor_name","vendor_vat_no","taxable_amount","freight_amount","delivery_challan_no"]
    # model = torch.load(model_path, map_location=device).to(device)
    # model = LayoutLMForTokenClassification.from_pretrained(model_path, use_safetensors=True)

    anno_label2idx = {k:v for v,k in enumerate(classes_info)}
    anno_idx2label = {k:v for k,v in enumerate(classes_info)}
    label2idx = {k:v for k,v in label2id.items()}
    idx2label = {i:x for x,i in label2idx.items()}
    print("Model Label ID mapping : ",label2idx, "\n\n\n")
    print("Model ID Label mapping : ",idx2label, "\n\n\n")
    # exit('+++++++++++++')

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['filename', 'field', 'actual_value', 'idp_predicted_value', 'fuzzy_score', 'box_predicted_value', 'confidence_score', 'bbox_actual', 'bbox_predicted', 'iou'])
        
        # Run inference
        t0 = time.time()
        # counter=100
        label_wise_fuzz75_pred_count = {} ; label_wise_fuzz100_pred_count = {} ; label_wise_fuzz85_pred_count={} ; label_wise_fuzz90_pred_count = {}; label_wise_fuzz100_pred_count = {} ; label_wise_total_pred_count = {}
        fuzz50_correct_preds = 0 ; fuzz25_correct_preds = 0 ; fuzz75_correct_preds = 0 ; fuzz85_correct_preds=0; fuzz90_correct_preds = 0 ; fuzz100_correct_preds = 0
        total_pred_labels = 0 ; total_actual_labels = 0

        for file in os.listdir(idp_inv_images_folder):
            # if file not in ["571581_Invoice_page_0.png"]: #["IM-000000010965506-AP_page_1.png"]:
            #     continue
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print('file:', file)
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            
            img_path = os.path.join(idp_inv_images_folder, file)
            img = cv2.imread(img_path)
            json_path = os.path.join(idp_inv_json_results, file.rsplit(".",1)[0] + ".json")
            try:
                with open(json_path,"r") as f:
                    idp_results = json.load(f)
            except:
                continue
            print(idp_results)
            raw_pred = list(idp_results.values())[0][idp_model_type]
            print("Model Prediction : ", raw_pred, "\n\n")
            pred = []
            for i in raw_pred["keys_bboxes"]:
                if isinstance(raw_pred["keys_bboxes"][i], list):
                    # print(raw_pred["keys_bboxes"][i], "=====")
                    for j in range(len(raw_pred["keys_bboxes"][i])):
                        total_pred_labels += 1
                        # print(j)
                        print(raw_pred["keys_bboxes"][i][j])
                        pred.append([raw_pred["keys_bboxes"][i][j], i, raw_pred["keys_confidence"][i][j], raw_pred["keys_extraction"][i][j]])    
                else:
                    total_pred_labels += 1
                    pred.append([raw_pred["keys_bboxes"][i], i, raw_pred["keys_confidence"][i], raw_pred["keys_extraction"][i]])

            # label_data = read_xml_label_data(f"{os.path.basename(img_path)[:-4]}.xml")
            print("Label File : ", f"{os.path.basename(img_path)[:-4]}.txt")
            label_data = read_text_label_data(f"{os.path.basename(img_path)[:-4]}.txt")
            # print("label_data : ", label_data)

            # for label_info in label_data: total_actual_labels += 1
            if itf_ocr:
                ocr_data = read_ocr_data(f"{os.path.basename(img_path)[:-4]}_text.txt")
                ocr_data = ocr_data.get('word_coordinates', [])
            else:
                ocr_data = read_ocr_data(f"{os.path.basename(img_path)[:-4]}_textAndCoordinates.txt")
            print("ocr_data : ", type(ocr_data))

            if ocr_data is None: exit()

            actual_value = get_actual_value(ocr_data, label_data, img.shape[1], img.shape[0], anno_idx2label)  # Pass image dimensions
            print("\n\nGot actual value : ", actual_value)
            total_actual_labels += len([x for inner_list in actual_value.values() for x in inner_list])

            # plot_boxes_on_image(img_path, pred, [x for inner_list in actual_value.values() for x in inner_list], idp_inv_image_results, file, plot_gt=plot_gt_flag)
            
            cls_to_delete = set()
            for xyxy, cls, cnf, idp_predicted_value in pred:
                print('$$$$$$$$$$$')
                print(xyxy, cls, cnf, idp_predicted_value)
                if cls != "O":
                    cls = cls[2:]
                print("#"*60)
                print(f"Prediction : {xyxy} | {cls} | {label2idx[cls]} | | {anno_label2idx[cls]}")
                print(f"Predicted Value : {idp_predicted_value}")
                actual_value_list = actual_value.get(anno_label2idx[cls], [{'words': 'N/A', 'bbox': (0,0,0,0)}])
                
                prediction_list = []
                for i, actual_value_str in enumerate(actual_value_list):
                    print("Actual value : ",actual_value_str)
                    
                    if anno_label2idx[cls] in actual_value.keys():
                        # print("Adding : ", cls, actual_value_str['words'])
                        cls_to_delete.add((cls, actual_value_str['words']))
                
                    bbox_predicted = xyxy
                    bbox_predicted = [int(coord) for coord in bbox_predicted]

                    predicted_value = merge_words_in_bbox(ocr_data, bbox_predicted)
                    print("Predicted value : ", predicted_value)
                    
                    iou = calculate_iou(bbox_predicted, actual_value_str["bbox"])
                    print("IOU : ", iou)
                    print("\n\n\n")

                    # ['filename', 'field', 'actual_value', 'idp_predicted_value', 'fuzzy_score', 'box_predicted_value', 'confidence_score', 'bbox_actual', 'bbox_predicted', 'iou']
                    idp_pred = idp_predicted_value
                    fuzz_score = calculate_fuzzy_score(actual_value_str['words'], idp_pred, cls)
                    
                    prediction_list.append([file, cls, actual_value_str['words'], idp_predicted_value, fuzz_score, predicted_value, round(cnf,3), actual_value_str['bbox'], bbox_predicted, round(iou,2)])

                final_pred = None ; max_fuzz = -1 ; max_iou = -1      
                final_pred = [file, cls, "N/A", idp_predicted_value, "0", predicted_value, round(cnf,3), "(0,0,0,0)", bbox_predicted, "0"]
                for pred in prediction_list:
                    if pred[-1] > max_iou:
                        final_pred = pred
                        max_fuzz = pred[4]
                        max_iou = pred[-1]
                    elif pred[4] > max_fuzz and pred[-1] > max_iou:
                        final_pred = pred
                        max_fuzz = pred[4]
                        max_iou = pred[-1]

                fuzzScore = max_fuzz

                print("!!!! fuzzScore :: ", fuzzScore)
                print("We picked : ", final_pred)
                print("prediction_list ::: ",prediction_list)
                if fuzzScore>=25: fuzz25_correct_preds += 1
                if fuzzScore>=50: fuzz50_correct_preds += 1 
                if fuzzScore>=100: 
                    fuzz100_correct_preds += 1
                    if cls not in label_wise_fuzz100_pred_count.keys():label_wise_fuzz100_pred_count[cls] = 1
                    else:label_wise_fuzz100_pred_count[cls] += 1
                if fuzzScore>=75: 
                    fuzz75_correct_preds += 1 
                    if cls not in label_wise_fuzz75_pred_count.keys():label_wise_fuzz75_pred_count[cls] = 1
                    else:label_wise_fuzz75_pred_count[cls] += 1
                if fuzzScore>=85: 
                    fuzz85_correct_preds += 1 
                    if cls not in label_wise_fuzz85_pred_count.keys():label_wise_fuzz85_pred_count[cls] = 1
                    else:label_wise_fuzz85_pred_count[cls] += 1
                if fuzzScore>=90: 
                    fuzz90_correct_preds += 1
                    if cls not in label_wise_fuzz90_pred_count.keys():label_wise_fuzz90_pred_count[cls] = 1
                    else:label_wise_fuzz90_pred_count[cls] += 1
                if fuzzScore>=100: 
                    fuzz90_correct_preds += 1
                    if cls not in label_wise_fuzz100_pred_count.keys():label_wise_fuzz100_pred_count[cls] = 1
                    else:label_wise_fuzz100_pred_count[cls] += 1
                
                if cls not in label_wise_total_pred_count.keys():label_wise_total_pred_count[cls] = 1
                else:label_wise_total_pred_count[cls] += 1
                
                # if fuzzScore >= 0:
                csv_writer.writerow(final_pred)
            
            for cls,actual_word in cls_to_delete:
                # print("Looking For : ", cls,actual_word)
                for i, word_info in enumerate(actual_value[anno_label2idx[cls]]):
                    if word_info['words'] == actual_word:
                        del actual_value[anno_label2idx[cls]][i]

            print("Actual Boxes Left : ", actual_value)
            for class_id, actual_box_left in actual_value.items():
                for actual_value_left_str in actual_box_left:
                    cls_ = anno_idx2label[class_id]
                    fuzzScore = 0 # calculate_fuzzy_score(actual_value_str['words'], "", cls_)
                    csv_writer.writerow([file, cls_, actual_value_left_str['words'], "N/A", fuzzScore, "N/A", 0, actual_value_left_str['bbox'], "(0,0,0,0)", 0]) #actual change 1
        print("###################################################")
        print("No. of Actual Labels :", total_actual_labels) 
        print("No. of Predictions   :", total_pred_labels) 
        precision_pt = 4
        print("Precision (fuzz>=25)  :", round(fuzz25_correct_preds/total_pred_labels,precision_pt))
        print("Precision (fuzz>=50)  :", round(fuzz50_correct_preds/total_pred_labels,precision_pt))
        print("Precision (fuzz>=75)  :", round(fuzz75_correct_preds/total_pred_labels,precision_pt))
        print("Precision (fuzz>=90)  :", round(fuzz90_correct_preds/total_pred_labels,precision_pt))
        print("Precision (fuzz>=100) :", round(fuzz100_correct_preds/total_pred_labels,precision_pt))
        print("Recall    (fuzz>=25)  :", round(fuzz25_correct_preds/total_actual_labels,precision_pt))
        print("Recall    (fuzz>=50)  :", round(fuzz50_correct_preds/total_actual_labels,precision_pt))
        print("Recall    (fuzz>=75)  :", round(fuzz75_correct_preds/total_actual_labels,precision_pt))
        print("Recall    (fuzz>=85)  :", round(fuzz85_correct_preds/total_actual_labels,precision_pt))
        print("Recall    (fuzz>=90)  :", round(fuzz90_correct_preds/total_actual_labels,precision_pt))
        print("Recall    (fuzz>=100) :", round(fuzz100_correct_preds/total_actual_labels,precision_pt))
        print("###################################################")
        print("report generation +++++++++++++++++++=")
        save_overall_geo_report(total_actual_labels, total_pred_labels, fuzz25_correct_preds, fuzz50_correct_preds, 
                        fuzz75_correct_preds, fuzz85_correct_preds, fuzz90_correct_preds, fuzz100_correct_preds)
        save_fuzzy_results(label_wise_total_pred_count, label_wise_fuzz75_pred_count, folder_name="geo_reports", filename="fuzzy75_results.xlsx")
        save_fuzzy_results(label_wise_total_pred_count, label_wise_fuzz100_pred_count, folder_name="geo_reports", filename="fuzzy100_results.xlsx")
        save_fuzzy_results(label_wise_total_pred_count, label_wise_fuzz90_pred_count, folder_name="geo_reports", filename="fuzzy90_results.xlsx")
        print("Label Wise Correct Predictions Percentage: (For Fuzzy percentage>=90)")
        for l in label_wise_total_pred_count.keys():
            if l not in label_wise_fuzz90_pred_count.keys():
                print(f">> Label {l}\t:\t0 / {label_wise_total_pred_count[l]}\t= 0 %")
            else:
                print(f">> Label {l}\t:\t{label_wise_fuzz90_pred_count[l]} / {label_wise_total_pred_count[l]}\t= {round(label_wise_fuzz90_pred_count[l]/label_wise_total_pred_count[l]*100,4)} %")
        print("###################################################")
        print("Label Wise Correct Predictions Percentage: (For Fuzzy percentage>=75)")
        for l in label_wise_total_pred_count.keys():
            if l not in label_wise_fuzz75_pred_count.keys():
                print(f">> Label {l}\t:\t0 / {label_wise_total_pred_count[l]}\t= 0 %")
            else:
                print(f">> Label {l}\t:\t{label_wise_fuzz75_pred_count[l]} / {label_wise_total_pred_count[l]}\t= {round(label_wise_fuzz75_pred_count[l]/label_wise_total_pred_count[l]*100,4)} %")
        print("###################################################")
        print("Label Wise Correct Predictions Percentage: (For Fuzzy percentage>=100)")
        for l in label_wise_total_pred_count.keys():
            if l not in label_wise_fuzz100_pred_count.keys():
                print(f">> Label {l}\t:\t0 / {label_wise_total_pred_count[l]}\t= 0 %")
            else:
                print(f">> Label {l}\t:\t{label_wise_fuzz100_pred_count[l]} / {label_wise_total_pred_count[l]}\t= {round(label_wise_fuzz100_pred_count[l]/label_wise_total_pred_count[l]*100,4)} %")
        
        if save_txt:
            print('Results saved to %s' % os.getcwd() + os.sep + out)

        print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=csv_file_path, help='output folder')  # output folder
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()