from datetime import datetime
import os
import time
import numpy as np
import pandas as pd
from torch.utils.data import random_split
from torch.optim import SGD, RMSprop
import torch
from PIL import Image,ImageSequence
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array3D, Array2D
from transformers import AdamW, LayoutLMForTokenClassification,LayoutLMTokenizer, LayoutLMConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import numpy as np
from functools import cmp_to_key
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import shutil
from seqeval.metrics import (
	classification_report)
from OLD.utility import get_logger_object_and_setting_the_loglevel, set_basic_config_for_logging
from tqdm import tqdm
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
from PIL import Image, ImageFont, ImageDraw  
from sklearn.cluster import DBSCAN
import cv2
import torchvision.transforms as transforms
from scipy.special import softmax
from configparser import ConfigParser
from nltk import ngrams
import re
from fuzzywuzzy import fuzz
import traceback

transform = transforms.ToTensor()
transform2 = transforms.ToPILImage()

configur = ConfigParser()
configur.read('train_valid.ini')


single_text_labels = [
	"bill_exchange_no",
	"bill_exchange_date",
	"boe_currency",
	"boe_amount",
	"country_of_origin",
	"invoice_no",
	"invoice_date",
	"invoice_currency",
	"invoice_amount",
	"tenor_type",
	"usance_tenor",
	"tenor_indicator",
	"indicator_type",
	"indicator_date",
	"invoice_due_date",
	"original_or_copy",
	"original_number",
	"lc_ref_no",
	"lc_date",
	"issue_place",
	"transaction_date",
	"awb_bill_no",
	"master_awb_bill_no",
	"house_awb_bill_no",
	"awb_bill_issue_date",
	"flight_no",
	"flight_date",
	"shipper_name",
	"shipper_country",
	"consignee_name",
	"consignee_country",
	"notify_party_name",
	"notify_party_country",
	"carrier_name",
	"agent_name",
	"place_of_receipt",
	"airport_of_departure",
	"airport_of_destination",
	"final_destination",
	"declared_value_of_carriage",
	"amount_insurance",
	"gross_quantity",
	"gross_weight",
	"net_weight",
	"good_marks",
	"invoice_number",
	"invoice_date",
	"lc_no",
	"lc_date",
	"freight_collect_or_prepaid",
	"freight_collected_at",
	"signed_by_carrier",
	"signed_by_agent",
	"awb_original_number",
	"awb_original_or_copy",
	"flight_details",
	"declared_value_of_custom",
	"transaction_date",
	"awb_bill_no",
	"master_awb_bill_no",
	"house_awb_bill_no",
	"awb_bill_issue_date",
	"flight_no",
	"flight_date",
	"shipper_name",
	"consignee_name",
	"notify_party_name",
	"notify_party_country",
	"carrier_name",
	"agent_name",
	"place_of_receipt",
	"airport_of_departure",
	"airport_of_destination",
	"final_destination",
	"declared_value_of_carriage",
	"amount_insurance",
	"gross_quantity",
	"gross_weight",
	"net_weight",
	"good_marks",
	"invoice_number",
	"invoice_date",
	"lc_no",
	"lc_date",
	"freight_collect_or_prepaid",
	"freight_collected_at",
	"awb_original_number",
	"awb_original_or_copy",
	"flight_details",
	"declared_value_of_custom",
	"at_place",	
]


vertical_merge_labels = ['shipper_country',
                         "shipper_address",
                         'carriage_condition',
                         'consignee_country',
                         'carrier_address', 'agent_address',
                         'goods_description', "description_of_goods",
                         'amount_in_words',
                         "drawee_bank_address",
                         "drawer_bank_address",
                         "drawee_address", "nostro_bank_address", "drawer_address",
                         "consignee_address", 'consignee_addres', 'shipper_address',
                         'notify_party_address', "consignor_address", "coo_issuer_address",
                         'nostro_bank_address', 'consignor_address', 'address_of_assured',
                         'drawee_address', 'insurance_issuer_address', 'remitter_address',
                         'beneficiary_address', 'coo_issuer_address', 'notify_party_address',
                         'drawer_bank_address', 'consignee_address', 'drawer_address',
                         'drawee_bank_address', 'insurance_issuer_address_bottom',
                         'drawer_bank_bottom_address', 'shipper_address',
                         'claim_payable_by_address', 'carrier_country', 'agent_country',"declaration","dimension",]

# used to order the field values inside a label. Refer Videos.
def contour_sort(a, b):
	if abs(a[1][1] - b[1][1]) <= 15:
		return a[1][0] - b[1][0]
	return a[1][1] - b[1][1]

def are_on_same_line(bbox1, bbox2, min_distance=0, tolerance=10):
    # Check if the vertical distance between the bottom of bbox1 and the top of bbox2 is within the tolerance
    # and if the overall distance is at least min_distance
    return (
        abs(bbox2[1] - bbox1[1]) <= tolerance
        and abs(bbox2[0] - bbox1[2]) >= min_distance
    )

def group_tokens_by_line(bbox_data, line_tolerance=5):
    all_bboxes_ = []
    line_wise_index = {}
    initial_bbox = []
    # Create sublists of OCR data in the same line with horizontal tolerance
    idx = 0
    for master_bbox in bbox_data:
        check_flag = False
        bbox = master_bbox
        if bbox not in all_bboxes_:
            all_bboxes_.append(bbox)
            line_wise_index[idx] = [bbox]
            check_flag = True

        if check_flag:
            prev_bbox = line_wise_index[idx][0]
            for single_bbox in bbox_data:
                if are_on_same_line(prev_bbox, single_bbox, tolerance=line_tolerance):
                    if single_bbox not in all_bboxes_:
                        line_wise_index[idx].append(single_bbox)
                        all_bboxes_.append(single_bbox)
            line_wise_index[idx].sort(key=lambda x: x[0])
            initial_bbox.append(line_wise_index[idx][0])
            idx += 1
    initial_bbox.sort(key=lambda x: x[1])
    updated_line_wise_data = []
    for bx in initial_bbox:
        for idx, values in line_wise_index.items():
            if bx in values:
                updated_line_wise_data.append(values)
    
    
    return updated_line_wise_data

def validate_contour_sort(word_bbox):
	all_bboxes = []
	for i in word_bbox:
		all_bboxes.append(i[-1])
	print(all_bboxes)
	final_word_bbox = []
	ordered_bbox = group_tokens_by_line(all_bboxes)
	for values in ordered_bbox:
		for val in values:
			for j in word_bbox:
				if j[-1] == val:
					final_word_bbox.extend([j])
	return final_word_bbox

def most_common(lst):
    return max(set(lst), key=lst.count)

def contour_sort(a, b):
    if abs(a[1][1] - b[1][1]) <= 15:
        return a[1][0] - b[1][0]
    return a[1][1] - b[1][1]


def get_iou_new(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {0, '2', 1, '3'}
		The (x1, 1) position is at the top left corner,
		the (2, 3) position is at the bottom right corner
	bb2 : dict
		Keys: {0, '2', 1, '3'}
		The (x, y) position is at the top left corner,
		the (2, 3) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou

def minimum_distance(bb1, bb2):
	# bb1 points
	min_distance = 9999999999
	p_11 = np.array((bb1[0], bb1[1]))
	p_12 = np.array((bb1[0], bb1[3]))
	p_13 = np.array((bb1[2], bb1[3]))
	p_14 = np.array((bb1[2], bb1[1]))
	all_points_bb1 = [p_11, p_12, p_13, p_14]
	# bb2 points
	p_21 = np.array((bb2[0], bb2[1]))
	p_22 = np.array((bb2[0], bb2[3]))
	p_23 = np.array((bb2[2], bb2[3]))
	p_24 = np.array((bb2[2], bb2[1]))
	all_points_bb2 = [p_21, p_22, p_23, p_24]
	for point1 in all_points_bb1:
		for point2 in all_points_bb2:
			dist = abs(np.linalg.norm(point1 - point2))
			if dist < min_distance:
				min_distance = dist
	return min_distance


def check_vertical_distribution(bb1, bb2):
	y1 = bb1[1]
	y2 = bb2[1]
	return abs(y1 - y2)


def model_output_sum(key, box, model_output):
	all_values = model_output[key]
	all_values = sorted(all_values, key=cmp_to_key(contour_sort))
	all_text = ""
	for value in all_values:
		try:
			iou = get_iou_new(value[1], box)
		except:
			continue
		if iou > 0:
			if all_text == "":
				all_text = value[0]
			else:
				all_text = all_text + " " + value[0]
	return all_text


def minimum_distance_vertical(bb1, bb2):
	# Calculate the minimum vertical distance between two bounding boxes
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

	min_distance_y = min(abs(y1_bb2 - y2_bb1), abs(y1_bb1 - y2_bb2))

	return min_distance_y

def get_iou_horizontal(bb1, bb2):
	# Calculate the Intersection over Union (IoU) of two bounding boxes (horizontal intersection)
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

	x_left = max(x1_bb1, x1_bb2)
	y_top = max(y1_bb1, y1_bb2)
	x_right = min(x2_bb1, x2_bb2)
	y_bottom = min(y2_bb1, y2_bb2)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
	bb1_area = (x2_bb1 - x1_bb1) * (y2_bb1 - y1_bb1)
	bb2_area = (x2_bb2 - x1_bb2) * (y2_bb2 - y1_bb2)
	union_area = bb1_area + bb2_area - intersection_area

	return intersection_area / union_area


def get_iou_vertical(bb1, bb2):
	# Calculate the Intersection over Union (IoU) of two bounding boxes (vertical intersection)
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

	x_left = max(x1_bb1, x1_bb2)
	y_top = max(y1_bb1, y1_bb2)
	x_right = min(x2_bb1, x2_bb2)
	y_bottom = min(y2_bb1, y2_bb2)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
	bb1_area = (x2_bb1 - x1_bb1) * (y2_bb1 - y1_bb1)
	bb2_area = (x2_bb2 - x1_bb2) * (y2_bb2 - y1_bb2)
	union_area = bb1_area + bb2_area - intersection_area

	return intersection_area / union_area


def get_intersection_percentage(bb1, bb2):
	# Calculate the percentage of vertical intersection between two bounding boxes
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

	x_left = max(x1_bb1, x1_bb2)
	y_top = max(y1_bb1, y1_bb2)
	x_right = min(x2_bb1, x2_bb2)
	y_bottom = min(y2_bb1, y2_bb2)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	intersection_area = max(0, y_bottom - y_top)
	bb1_area = y2_bb1 - y1_bb1
	bb2_area = y2_bb2 - y1_bb2

	return intersection_area / min(bb1_area, bb2_area)

def special_chr_check(bb_token, flag):
	if ',' in bb_token:
		flag = False
	if '-' in bb_token:
		flag = False
	return flag

def area(coordinates):
	l = coordinates[2] - coordinates[0]
	h = coordinates[3] - coordinates[1]
	return l * h

def merge_surrounding(data, model_output, w, h):
	new = data.copy()
	print('entered into merging_surroundings ++++++++++++++++++++++++++++++++++++++')
	for key in list(data.keys()):
		print(key)
		if key in vertical_merge_labels:
			all_values = data[key]
			# all_values = vertical_horizontal_values(new_all_values)
			# if key=='drawee_address':
			#     print(all_values)
			#     exit("PPPPPPPPPPPP")
			bboxes = [x[1] for x in data[key]]
			eps_horizontal = 100  # Threshold for horizontal merging
			eps_vertical = 100  # Threshold for vertical merging
			######################
			# if w>h:
			#     eps_horizontal = round(h*17/100)#100
			#     eps_vertical = round(w*12/100) #100
			# else:
			#     eps_horizontal = round(h*12/100)#100
			#     eps_vertical = round(w*17/100) #100
			############################
			all_values = data[key]
			print(all_values)
			# for all_values in new_all_values:
			length = len(all_values)
			if length > 1:
				i = 0
				# if (bb1_height > bb1_width and bb2_height > bb2_width) or (bb1_height < bb1_width and bb2_height < bb2_width):
				while i in range(length - 1):
					print(i)
					bb1 = all_values[i][1]
					bb2 = all_values[i + 1][1]
					bb1_token = all_values[i][0]
					bb2_token = all_values[i + 1][0]
					min_dist_horizontal = minimum_distance(bb1, bb2)
					min_dist_vertical = minimum_distance_vertical(bb1, bb2)

					try:
						IOU_horizontal = get_iou_horizontal(bb1, bb2)
						IOU_vertical = get_iou_vertical(bb1, bb2)
						inter_percentage = get_intersection_percentage(bb1, bb2)
					except:
						i = i + 1
						continue
					bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1
					bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2
					bb1_width = bb1_x2 - bb1_x1
					bb1_height = bb1_y2 - bb1_y1
					bb2_width = bb2_x2 - bb2_x1
					bb2_height = bb2_y2 - bb2_y1
					# if len(bb1_token)<3:
					print('beore', bb2_width, bb2_height)
					print('bb2_token length', len(bb2_token))
					flag1 = True
					flag2 = True
					flag1 = special_chr_check(bb1_token, flag1)
					flag2 = special_chr_check(bb2_token, flag2)

					if len(bb1_token) == 1 or (len(bb1_token) < 3 and flag1 == False):
						temp = bb1_width
						bb1_width = bb1_height
						bb1_height = temp
					if len(bb2_token) == 1 or (len(bb2_token) < 3 and flag2 == False):
						temp = bb2_width
						bb2_width = bb2_height
						bb2_height = temp
					print('flag2', flag2)
					print('bb2_token', bb2_token)
					print('bb1_width', bb1_width, 'bb1_height', bb1_height)
					print('bb2_width', bb2_width, 'bb2_height', bb2_height)
     
					# if (bb1_height >= bb1_width and bb2_height >= bb2_width) or (
					# 		bb1_height <= bb1_width and bb2_height <= bb2_width):
     
					print('entered into first if')
					if (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) or (
							min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage):
						print('entered into second if')
						print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
						x_left = min(bb1[0], bb2[0])
						y_top = min(bb1[1], bb2[1])
						x_right = max(bb1[2], bb2[2])
						y_bottom = max(bb1[3], bb2[3])
						box = [x_left, y_top, x_right, y_bottom]
						text = model_output_sum(key, box, model_output)
						print("merged text is ", text)
						new_value = [text, box]
						print(new_value)
						all_values.remove(all_values[i])
						all_values.remove(all_values[i])
						all_values.insert(i, new_value)
						print(all_values)
						length = len(all_values)
						if length == 1:
							print("will break")
							break
						# else:
						# 	print("distance is very high")
						# 	i = i + 1
					else:
						i = i + 1
			else:
				print("will continue")
				continue


		else:
			print('Vertical merging not happening ++++++++++++++++++++++++++++++++++++')
			print(key)
			bboxes = [x[1] for x in data[key]]
			if w > h:
				v_eps = round(h * 1.5 / 100)  # 10round(number)
				h_eps = round(w * 5 / 100)  # 36
			else:
				v_eps = round(h * 1.1 / 100)  # 10round(number)
				h_eps = round(w * 5.8 / 100)  # 36
			all_values = data[key]
			print('all_values >>>>>>>>>>>', all_values)
			length = len(all_values)
			if length > 1:
				i = 0
				while i in range(length - 1):
					print(i)
					bb1 = all_values[i][1]
					bb2 = all_values[i + 1][1]
					# ocr_confs = [all_values[i][3],all_values[i+1][3]]
					# min_dist = minimum_distance(bb1, bb2)
					vertical_distance = check_vertical_distribution(bb1, bb2)
					# dist_bwt_words = (abs(bb1[2]-bb2[0])/w)*100
					hori_distance = abs(bb1[2] - bb2[0])
					print('min_dist========>', vertical_distance, 'bb1', bb1, 'bb2', bb2)
					print('hori_distance ==========>', hori_distance)
					try:
						IOU = get_iou_new(bb1, bb2)
						print('IOU>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', IOU)
					except:
						i = i + 1
						continue
					if (vertical_distance <= v_eps and hori_distance < h_eps) or IOU > 0.1:
						print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
						x_left = min(bb1[0], bb2[0])
						y_top = min(bb1[1], bb2[1])
						x_right = max(bb1[2], bb2[2])
						y_bottom = max(bb1[3], bb2[3])
						box = [x_left, y_top, x_right, y_bottom]
						text = model_output_sum(key, box, model_output)
						print("merged text is ", text)
						"""if "NA" in ocr_confs:
                            avg_ocr_confs = "NA"
                        else:
                            avg_ocr_confs = ( ocr_confs[0]* area(bb1) + ocr_confs[1]*area(bb2) )/(area(bb1) + area(bb2))"""
						new_value = [text, box]
						print(new_value)
						all_values.remove(all_values[i])
						all_values.remove(all_values[i])
						all_values.insert(i, new_value)
						print(all_values)
						length = len(all_values)
						if length == 1:
							print("will break")
							break
					else:
						print("distance is very high")
						i = i + 1
			else:
				print("will continue")
				continue
		# if key=="shipper_address":
		#     print('True')
		#     # print(all_values)
		#     exit('+++++++++++++++++=')



def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]


# Normalizes all points after zooming
def normalize(points: list, width: int, height: int) -> list:
	x0, y0, x2, y2 = [int(p) for p in points]

	x0 = int(1000 * (x0 / width))
	x2 = int(1000 * (x2 / width))
	y0 = int(1000 * (y0 / height))
	y2 = int(1000 * (y2 / height))

	if x0 > 1000:
		x0 = 1000
	# print(">")
	if x0 < 0:
		x0 = 0
	# print("<")
	if x2 > 1000:
		x2 = 1000
	# print(">")
	if x2 < 0:
		x2 = 0
	# print("<")
	if y0 > 1000:
		y0 = 1000
	# print(">")
	if y0 < 0:
		y0 = 0
	# print("<")
	if y2 > 1000:
		y2 = 1000
	# print(">")
	if y2 < 0:
		y2 = 0
	# print("<")
	return [x0, y0, x2, y2]


def tuple_to_string(sen):
	str_test = sen
	word = ""
	for i, w in enumerate(str_test):
		if i == 0:
			word = word + w
		else:
			word = word + ' ' + w
	return word


def lookup(
		text,
		n_words,
		match_threshold,
		file_path,
		result_set,
		key
):
	try:
		out_dict = {}
		out_list = []
		all_box = []
		message = ""

		file = open(file_path, "r")
		lines = file.readlines()

		txt_words = []
		for l in lines:
			line = l.split("\n")
			txt_words.append(line[0])

		if n_words > 5:
			status = "N words larger than 5, provide N words less than 5"
			out_dict["status"] = status
			return out_dict

		if match_threshold < 80:
			status = "Matching threshold value less than 80, provide Matching threshold greater than 80"
			out_dict["status"] = status
			return out_dict
		else:
			res = re.sub(r"[^\w\s]", "", text)
			gram_list = []
			for j in range(n_words):
				gram_count = ngrams(res.split(), j + 1)

				for gram in gram_count:
					sen = tuple_to_string(gram)
					gram_list.append(sen)

			for word in gram_list:
				for txt_char in txt_words:

					# print(word[0].lower())

					if fuzz.ratio(word.lower(), txt_char) > match_threshold:
						# if word.lower() in txt_words:
						# print(word, "----", txt_char, fuzz.ratio(word.lower(), txt_char))

						info_dict = {}
						info_dict["searched_string"] = word  # searched string is our data.
						info_dict["found_string"] = txt_char  # found string is present in countries.txt (lookup file)
						info_dict["string_match_value"] = fuzz.ratio(
							word.lower(), txt_char
						)
						out_list.append(info_dict)
			if len(out_list) != 0:
				for res in out_list:
					look_up = res['found_string']
					# print(found)
					original = res['searched_string']
					original_words = original.split()
					for val in result_set[key]:
						for word in original_words:
							if fuzz.ratio(word.lower(), val[0]) > match_threshold:
								all_box.append(val[1])
					x1 = min([x[0] for x in all_box])
					x2 = max([x[2] for x in all_box])
					y1 = min([x[1] for x in all_box])
					y2 = max([x[3] for x in all_box])
					box_result = [x1, y1, x2, y2]
					res['bbox'] = box_result


			return out_list
		file.close()   
	except Exception as e:
		print(traceback.format_exc())



if __name__=="__main__":
	model = LayoutLMForTokenClassification.from_pretrained("/home/ng6309/datascience/hridesh/layoutlmv1/extraction/AWB/Best_Model")
	model.to(device)

	tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

	labels = ['S-carriage_condition', 'S-flight_date', 'S-consignee_name', 'S-airport_of_departure', 'S-shipper_name', 'S-signed_by_agent', 'S-signed_by_carrier', 'S-agent_name', 'S-notify_party_name', 'S-at_place', 'S-master_awb_bill_no', 'S-flight_no', 'S-dimension', 'S-final_destination', 'S-place_of_receipt', 'S-lc_no', 'S-gross_quantity', 'S-agent_address', 'S-amount_insurance', 'S-net_weight', 'S-consignee_address', 'S-notify_party_address', 'S-freight_collect_or_prepaid', 'S-awb_bill_no', 'S-invoice_number', 'S-awb_original_or_copy', 'S-invoice_date', 'S-carrier_address', 'S-good_marks', 'O', 'S-carrier_name', 'S-declared_value_of_carriage', 'S-shipper_address', 'S-lc_date', 'S-airport_of_destination', 'S-awb_original_number', 'S-goods_description', 'S-house_awb_bill_no', 'S-transaction_date', 'S-flight_details', 'S-freight_collected_at', 'S-awb_bill_issue_date', 'S-declared_value_of_custom', 'S-gross_weight']

	id2label = {v: k for v, k in enumerate(labels)}
	query_index = 0

	input_folder = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/AWB_EVL"
	image_folder = input_folder+"/Images"
	label_folder = input_folder+"/Labels"
	ocr_path = input_folder+"/OCR"
	label_path = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/AWB/classes.txt"
	output_path = input_folder+"/results"

	os.makedirs(output_path,exist_ok=True)

	with open(label_path, "r") as f:
		labels = eval(f.read()) 
	f.close()

	labels = [x.replace("S-", "") for x in labels]
	label2id = {label: idx for idx, label in enumerate(labels)}
	id2label = {idx: label for idx, label in enumerate(labels)}

	for image in os.listdir(image_folder):
		image_path = os.path.join(image_folder,image)
		print("image_path:  ",image_path)
		im = Image.open(image_path)
		directory, base_name = os.path.split(image_path)
		base_name_without_ext, _ = os.path.splitext(base_name)
		ocr_json_path = os.path.join(ocr_path,base_name_without_ext+".json")
		count = 0
		all_page_result = {}
		for i, image in enumerate(ImageSequence.Iterator(im)):
			count += 1
			w, h = image.size
			temp = image.convert("L")
			image_data = np.asarray(temp)
			image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
			arr = transform(image)
			if os.path.exists(ocr_json_path):
				json_ocr = json.load(open(os.path.join(ocr_json_path), "r"))
				all_text, word_coordinates = json_ocr["all_text"],json_ocr["word_coordinates"]
			
			words: list = []
			bboxes: list = []
			bounding_boxes: list = []


			for t in word_coordinates:
				if 'right' in list(t.keys()):
					t['x1'] = t['left']
					t['y1'] = t['top']
					t['x2'] = t['right']
					t['y2'] = t['bottom']
				words.append(t['word'])
				bounding_boxes.append([t['x1'], t['y1'], t['x2'], t['y2']])
				bboxes.append(normalize([t['x1'], t['y1'], t['x2'], t['y2']], w, h))

			token_boxes=[]

			for word, box in zip(words, bboxes):
				word_tokens = tokenizer.tokenize(word)
				token_boxes.extend([box] * len(word_tokens))

			token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

			encoded_inputs = tokenizer(" ".join(words), return_tensors="pt")
			
			bbox = torch.tensor([token_boxes])
			encoded_inputs["bbox"] = bbox

			input_id_chunks = list(encoded_inputs['input_ids'][0].split(300))
			token_type_id_chunks = list(encoded_inputs['token_type_ids'][0].split(300))
			mask_chunks = list(encoded_inputs['attention_mask'][0].split(300))
			bbox_chunks = list(encoded_inputs['bbox'][0].split(300))
			image_chunks = list()
			for i in range(len(input_id_chunks)):
				input_id_chunks[i] = torch.cat([torch.tensor([101]), input_id_chunks[i], torch.tensor([102])])
				token_type_id_chunks[i] = torch.cat([torch.tensor([0]), token_type_id_chunks[i], torch.tensor([0])])
				mask_chunks[i] = torch.cat([torch.tensor([1]), mask_chunks[i], torch.tensor([1])])
				bbox_chunks[i] = torch.cat([torch.tensor([[0, 0, 0, 0]]), bbox_chunks[i], torch.tensor([[0, 0, 0, 0]])])
				pad_len = 512 - input_id_chunks[i].shape[0]
				if pad_len > 0:
					input_id_chunks[i] = torch.cat([input_id_chunks[i], torch.Tensor([0] * pad_len)])
					token_type_id_chunks[i] = torch.cat([token_type_id_chunks[i], torch.Tensor([0] * pad_len)])
					mask_chunks[i] = torch.cat([mask_chunks[i], torch.Tensor([0] * pad_len)])
					bbox_chunks[i] = torch.cat([bbox_chunks[i], torch.Tensor([[0, 0, 0, 0]] * pad_len)])



			input_ids = torch.stack(input_id_chunks)
			attention_mask = torch.stack(mask_chunks)
			token_type_ids = torch.stack(token_type_id_chunks)
			bbox = torch.stack(bbox_chunks)

			input_dict = {
            'input_ids': input_ids.long().to(device),
            'attention_mask': attention_mask.float().to(device),
            'token_type_ids': token_type_ids.long().to(device),
            'bbox': bbox.long().to(device),
        	}

			outputs = model(**input_dict)


			all_predictions: list = []
			all_boxes: list = []
			all_confidences: list = []
			all_text: list = []


			for i, output in enumerate(outputs.logits):
				#print(i, output)
				# converting back into PIL image
				new_img = transform2(arr)
				# loading the image font
				
				predictions = output.argmax(-1).squeeze().tolist()
				#print('the predictions', predictions)
				
				confidences = softmax(output.cpu().detach().numpy(), axis=1)
				#print(confidences)
				
				max_confidences = np.max(confidences, axis=1).reshape(confidences.shape[0], -1)
				#print(max_confidences)
				
				all_confidences += [x[0] for x in max_confidences]
				#print(all_confidences)
				
				token_boxes = bbox_chunks[i].squeeze().tolist()
				width, height = new_img.size
				true_predictions = [id2label[prediction] for prediction in predictions]
				all_predictions += true_predictions
				#print('all_predictions',all_predictions)
				
				true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
				all_boxes += true_boxes
				#print(all_boxes)
				c=0
				for id in input_dict['input_ids'][i]:
					all_text.append(tokenizer.decode([id]))

			del outputs
			new_img = transform2(arr)
			draw = ImageDraw.Draw(new_img)

			curr_box: list = []
			results_pred: list = []
			results_bbox: list = []
			results_text: list = []
			temp_preds: list = [] 
			temp_text: list = []
			sep_index: list = all_text.index('[SEP]')

			if len(all_text) > 512:
				if '[PAD]' in all_text:
					sep_index = all_text.index('[PAD]') - 2
				else:
					sep_index = len(all_text) - 3


			for i in range(len(all_text)):
				if all_text[i] not in ['[CLS]', '[SEP]', '[PAD]']:  # and all_predictions[i] != 'O':
					# print(i)
					if (curr_box != all_boxes[i] and \
						len(temp_text) > 0) or \
							(i == sep_index - 1 and len(temp_text) > 0):
						
						# print("1: ", all_text[i])
						if i == sep_index - 1:
							temp_text.append(all_text[i])
							temp_confs.append(all_confidences[i])
							temp_preds.append(all_predictions[i])
						text = ""
						pred = "O"
						preds = [x for x in temp_preds if x != 'O']
						conf = 0
						if len(preds) > 0:
							pred = most_common(preds)
							
						for j in range(len(temp_text)):
							text += temp_text[j].replace("##", "")
						

						results_text.append(text)
						results_pred.append(pred)
						results_bbox.append(curr_box)

						temp_text = []
						temp_confs = []
						temp_preds = []
						if i != sep_index - 1:
							temp_text.append(all_text[i])
							temp_confs.append(all_confidences[i])
							temp_preds.append(all_predictions[i])
							curr_box = all_boxes[i]
					elif curr_box == all_boxes[i]:
						temp_text.append(all_text[i])
						temp_preds.append(all_predictions[i])
					elif len(temp_text) == 0:
						temp_text.append(all_text[i])
						temp_preds.append(all_predictions[i])
						curr_box = all_boxes[i]

			result_set = {}

			for i in range(len(results_pred)):
				if results_pred[i] != 'O':
					if results_pred[i] not in list(result_set.keys()):
						result_set[results_pred[i]] = []
					result_set[results_pred[i]].append([results_text[i], 
														results_bbox[i]])

			model_output = result_set.copy()
			with open(os.path.join(output_path, base_name_without_ext + "model_output.txt"), "w") as f:
				json.dump(result_set, f)
			final_result_set = {}
			f.close()


			for k in list(result_set.keys()):
				if k not in single_text_labels:
					try:
						alpha = float(configur[k]['ALPHA'])
					except:
						alpha = float(configur['Default']['ALPHA'])
					if len(result_set[k]) > 1:
						texts = [x[0] for x in result_set[k]]
						bboxes = [x[1] for x in result_set[k]]
						avg_w = np.mean([abs(x[0] - x[2]) for x in bboxes])
						avg_h = np.mean([abs(x[1] - x[3]) for x in bboxes])
						eps = np.sqrt(avg_w ** 2 + avg_h ** 2) * alpha
						# if eps<=0.0:
						# 	eps=0.1
						clustering = DBSCAN(eps=eps, min_samples=1).fit(bboxes)
						label_set = set(clustering.labels_)
						for l in label_set:
							selected = list(np.where(clustering.labels_ == l)[0])
							selected_texts = [x for i, x in enumerate(texts) if i in selected]
							selected_boxes = [x for i, x in enumerate(bboxes) if i in selected]
							text_boxes = [[x, y] for x, y in zip(selected_texts, selected_boxes)]
							text_boxes = sorted(text_boxes, key=cmp_to_key(contour_sort))
							text_result = ""
							for tb in text_boxes:
								if text_result == "":
									text_result += tb[0]
								else:
									text_result += " " + tb[0]
							# print(text_result)
							x1 = min([x[0] for x in selected_boxes])
							x2 = max([x[2] for x in selected_boxes])
							y1 = min([x[1] for x in selected_boxes])
							y2 = max([x[3] for x in selected_boxes])
							box_result = [x1, y1, x2, y2]
							# print(box_result)
							if k not in list(final_result_set.keys()):
								final_result_set[k] = []
							final_result_set[k].append([text_result, box_result])
					else:
						if k not in list(final_result_set.keys()):
							final_result_set[k] = []
						final_result_set[k].append([result_set[k][0][0],result_set[k][0][1]])
				else:
						if len(result_set[k]) > 1:
							texts = [x[0] for x in result_set[k]]
							bboxes = [x[1] for x in result_set[k]]
							for i, value in enumerate(zip(texts, bboxes)):
								if k not in list(final_result_set.keys()):
									final_result_set[k] = []
								final_result_set[k].append(list(value))
						else:
							if k not in list(final_result_set.keys()):
								final_result_set[k] = []
							final_result_set[k].append([result_set[k][0][0],result_set[k][0][1]])  

			merge_surrounding(final_result_set, model_output, w, h)

			print("final_result_set:#####################################################")
			print(final_result_set)

			for k in list(final_result_set.keys()):
				all_values = final_result_set[k]
				# exit('________________')
				# for value in all_values:
				#     draw.rectangle(value[1], outline=label2color[k], width=2)
				#     draw.text((value[1][0] + 5, value[1][1] - 20),
				#                 text=k , fill=label2color[k], font=font)
				for value in all_values:
					draw.rectangle(value[1], outline="red", width=2)
					draw.text(
								(value[1][0] + 5, value[1][1] - 20),
								text=k, 
								fill="blue")
			lookup_result = {}
			all_page_result["Page Number " + str(count)] = final_result_set
			for k in list(final_result_set.keys()):
				if k in ["applicant_country", "beneficiary_country"]:
					for val in final_result_set[k]:
						result_country = lookup(val[0], 4, 90, "countries.txt", result_set, k)
						result_company = lookup(val[0], 4, 90, "organization.txt", result_set, k)
						for res in result_company:
							found = res['found_string']
							searched = res['searched_string']
							new_val = val[0].replace(searched, found)
							val[0] = new_val
						# replacing ocr result with correct result
						for res in result_country:
							found = res['found_string']
							searched = res['searched_string']
							new_val = val[0].replace(searched, found)
							val[0] = new_val
						for res in result_country:
							if (str(k) + "-country") not in lookup_result:
								lookup_result[('LUT_' + str(k) + "-country")] = []
							lookup_result[('LUT_' + str(k) + "-country")].append(
								(res['found_string'], res['string_match_value'], res["bbox"]))
						for res in result_company:
							if (str(k) + "-organization") not in lookup_result:
								lookup_result[('LUT_' + str(k) + "-organization")] = []
							lookup_result[('LUT_' + str(k) + "-organization")].append(
								(res['found_string'], res['string_match_value'], res["bbox"]))



			with open(os.path.join(output_path, base_name_without_ext + ".json"), "w") as f:
				json.dump(final_result_set, f)
			f.close()

			new_img.save(os.path.join(output_path, base_name_without_ext + ".png"))
		with open(os.path.join(output_path, base_name_without_ext + "all_page.json"), "w") as f:
				json.dump(all_page_result, f)
				f.close()
