from sklearn.cluster import DBSCAN
import numpy as np
import os 
import json
import configparser
from functools import cmp_to_key
import random
from PIL import Image, ImageDraw, ImageFont
import re
from PIL import ImageSequence
import cv2


App_Filepath = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
# config.read(App_Filepath + '/config.ini')
config.read('/home/gpu1admin/rakesh/geo-layout-lm-tf-Final_geo_training_inference_scripts/config.ini')

import numpy as np
from sklearn.cluster import DBSCAN
from functools import cmp_to_key
import json
import traceback
from nltk import ngrams
import re
from scipy.special import softmax
from fuzzywuzzy import fuzz
import psutil
import torchvision.transforms as transforms

transform2 = transforms.ToPILImage()
transform = transforms.ToTensor()
 # provide the path of vertical merging_labels (Note:The files will get in data folder)
merge_models_path= "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/vertical_merging_labels"     

# provide the path of single_text_labels (Note:The files will get in data folder)
single_text_path= "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/single_text_labels"

single_text_labels=[]

vertical_merge_labels= []

# with open(merge_models_path) as f:
# 	vertical_merge_labels=[label.strip() for label in f.readlines()]

# with open(single_text_path) as f:
# 	single_text_labels=[label.strip() for label in f.readlines()]


def contour_sort(a, b):
	if abs(a[1][1] - b[1][1]) <= 15:
		return a[1][0] - b[1][0]
	return a[1][1] - b[1][1]
def minimum_distance_horizontal(bb1, bb2):
    # Calculate the minimum horizontal distance between two bounding boxes
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    min_distance_x = min(abs(x1_bb2 - x2_bb1), abs(x1_bb1 - x2_bb2))
    min_distance_y = min(abs(y1_bb1 - y1_bb2), abs(y2_bb1 - y2_bb2))

    return min(min_distance_x, min_distance_y)

# def minimum_distance_vertical(bb1, bb2):
#     # Calculate the minimum vertical distance between two bounding boxes
#     x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
#     x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

#     min_distance_x = min(abs(x1_bb1 - x1_bb2), abs(x2_bb1 - x2_bb2))
#     min_distance_y = min(abs(y1_bb2 - y2_bb1), abs(y1_bb1 - y2_bb2))

#     return min(min_distance_x, min_distance_y)
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

def calculate_bounding_box_area(box_coords):
    """
    Calculate the area of a bounding box.

    Parameters:
    box_coords (list): List containing the coordinates of the bounding box in the format [x_min, y_min, x_max, y_max].

    Returns:
    int: The area of the bounding box.
    """
    x_min, y_min, x_max, y_max = box_coords
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    return area

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

def area(coordinates):
	l = coordinates[2] - coordinates[0]
	h = coordinates[3] - coordinates[1]
	return l * h

def merge_surrounding(data, model_output):
	new = data.copy()
	print('entered into merging_surroundings ++++++++++++++++++++++++++++++++++++++')
	for key in list(data.keys()):
		print(key)
		if key in vertical_merge_labels:
			bboxes = [x[1] for x in data[key]]
			eps_horizontal = 100  # Threshold for horizontal merging
			eps_vertical = 100    # Threshold for vertical merging
			all_values = data[key]
			print(all_values)
			length = len(all_values)
			if length > 1:
				i = 0
				while i in range(length - 1):
					print(i)
					bb1 = all_values[i][1]
					bb2 = all_values[i + 1][1]
					# confs = [all_values[i][2], all_values[i + 1][2]]
					min_dist_horizontal = minimum_distance(bb1, bb2)
					min_dist_vertical = minimum_distance_vertical(bb1, bb2)

					try:
						IOU_horizontal = get_iou_horizontal(bb1, bb2)
						IOU_vertical = get_iou_vertical(bb1, bb2)
						inter_percentage = get_intersection_percentage(bb1, bb2)
					except:
						i = i + 1
						continue
					if (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage>0) or (min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage):
						print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
						x_left = min(bb1[0], bb2[0])
						y_top = min(bb1[1], bb2[1])
						x_right = max(bb1[2], bb2[2])
						y_bottom = max(bb1[3], bb2[3])
						box = [x_left, y_top, x_right, y_bottom]
						text = model_output_sum(key, box, model_output)
						print("merged text is ", text)
						# avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
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
		else:
			print('Vertical merging not happening ++++++++++++++++++++++++++++++++++++')
			print(key)
			bboxes = [x[1] for x in data[key]]
			eps = 100
			all_values = data[key]
			print(all_values)
			length = len(all_values)
			if length > 1:
				i = 0
				while i in range(length - 1):
					print(i)
					bb1 = all_values[i][1]
					bb2 = all_values[i + 1][1]
					# confs = [all_values[i][2], all_values[i + 1][2]]
					# ocr_confs = [all_values[i][3],all_values[i+1][3]]
					min_dist = minimum_distance(bb1, bb2)
					try:
						IOU = get_iou_new(bb1, bb2)
					except:
						i = i + 1
						continue
					if min_dist <= eps or IOU > 0:
						print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
						x_left = min(bb1[0], bb2[0])
						y_top = min(bb1[1], bb2[1])
						x_right = max(bb1[2], bb2[2])
						y_bottom = max(bb1[3], bb2[3])
						box = [x_left, y_top, x_right, y_bottom]
						text = model_output_sum(key, box, model_output)
						print("merged text is ", text)
						# avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
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
# def merge_surrounding(data, model_output):
# 	new = data.copy()
# 	print('entered into merging_surroundings ++++++++++++++++++++++++++++++++++++++')
# 	for key in list(data.keys()):
# 		print(key)
# 		bboxes = [x[1] for x in data[key]]
# 		eps_horizontal = 100  # Threshold for horizontal merging
# 		eps_vertical = 100    # Threshold for vertical merging
# 		all_values = data[key]
# 		print(all_values)
# 		length = len(all_values)
# 		if length > 1:
# 			i = 0
# 			while i in range(length - 1):
# 				print(i)
# 				bb1 = all_values[i][1]
# 				bb2 = all_values[i + 1][1]
# 				# confs = [all_values[i][2], all_values[i + 1][2]]
# 				min_dist_horizontal = minimum_distance(bb1, bb2)
# 				min_dist_vertical = minimum_distance_vertical(bb1, bb2)

# 				try:
# 					IOU_horizontal = get_iou_horizontal(bb1, bb2)
# 					IOU_vertical = get_iou_vertical(bb1, bb2)
# 					inter_percentage = get_intersection_percentage(bb1, bb2)
# 				except:
# 					i = i + 1
# 					continue
# 				if (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage>0) or (min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage):
# 					print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
# 					x_left = min(bb1[0], bb2[0])
# 					y_top = min(bb1[1], bb2[1])
# 					x_right = max(bb1[2], bb2[2])
# 					y_bottom = max(bb1[3], bb2[3])
# 					box = [x_left, y_top, x_right, y_bottom]
# 					text = model_output_sum(key, box, model_output)
# 					print("merged text is ", text)
# 					# avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
# 					new_value = [text, box]
# 					print(new_value)
# 					all_values.remove(all_values[i])
# 					all_values.remove(all_values[i])
# 					all_values.insert(i, new_value)
# 					print(all_values)
# 					length = len(all_values)
# 					if length == 1:
# 						print("will break")
# 						break
# 				else:
# 					print("distance is very high")
# 					i = i + 1
# 		else:
# 			print("will continue")
# 			continue


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

			# fourgrams = ngrams(res.split(), n)

			for j in range(n_words):
				gram_count = ngrams(res.split(), j + 1)

				# ourgrams = ngrams(res.split(), n)

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

			print(out_list)
			# out_dict['response'] = out_list
			# out_dict['status'] = 'success'
			# print(out_dict)

			return out_list
	except Exception as e:
		print(traceback.format_exc())

input_path= "/home/gpu1admin/rakesh/geo_testing/data_oct21/data_in_funsd_format/dataset/custom_trial"
file_path= os.path.join(input_path,'Results/annotations')
result_path = os.path.join(input_path, "Results_PL_validated")
if not os.path.exists(result_path):
        os.mkdir(result_path)
img_path= os.path.join(input_path, "Results/images")
image_list= os.listdir(file_path)
image_list=  [ image.split('_tagging.json')[0] for image in image_list if image.endswith(".json")]
# image_list= 
print(image_list)
number_of_colors = 80
color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

# exit('+++++++++++++++++++')
count=0
for file in image_list:
	print(file)
	count+=1
	im = Image.open(os.path.join(img_path, file+"_linking.png"))
	for i, image in enumerate(ImageSequence.Iterator(im)):
		w, h = image.size
		temp = image.convert("L")
		image_data = np.asarray(temp)
		image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
		# plt.imshow(image)
		# plt.show()
		# plt.close()
		arr = transform(image)
		try:
			with open(os.path.join(file_path,file+"_tagging.json")) as f:
				token_data = json.load(f)['form']
		except Exception as e:
			with open(os.path.join(input_path,file+"_tagging.json")) as f:
				token_data = json.load(f)
		predicted_keys = [token["pred_key"] for token in token_data if not token["pred_key"]=='O']
		labels= [token.split('-')[1]for token in predicted_keys]
		labels= set(labels)
		label2color = {}
		for i, l in enumerate(labels):
			label2color[l] = color[i]
		font = ImageFont.truetype("/home/gpu1admin/rakesh/geo-layout-lm-tf-Final_geo_training_inference_scripts/arial.ttf", 20)
		# Extract predicted keys and corresponding coordinates
		coordinates = [token["coords"] for token in token_data if not token["pred_key"]=='O']
		new_img = transform2(arr)
		# new_img.show()
		# exit('++++++++====')
		draw = ImageDraw.Draw(new_img)
		# print(len(predicted_keys))
		# print(predicted_keys)
		# print(len(coordinates))
		# print(token_data)
		# exit('+++++++++++++++++++')
		result_set = {}
		for i in range(len(token_data)):
				if token_data[i]["pred_key"]!= 'O':
					if (token_data[i]['pred_key']).split('-')[1] not in list(result_set.keys()):
						result_set[(token_data[i]['pred_key']).split('-')[1]] = []
					result_set[(token_data[i]['pred_key']).split('-')[1]].append([token_data[i]['text'], token_data[i]['coords']])
		print(result_set)
		model_output = result_set.copy()
		final_result_set = {}
		for k in list(result_set.keys()):
			if k not in single_text_labels:
				try:
					alpha = float(config[k]['ALPHA'])
				except:
					alpha = float(config['Default']['ALPHA'])
				if len(result_set[k]) > 1:
					print("++++++++++++++entry in this block+++++++++++")
					texts = [x[0] for x in result_set[k]]
					bboxes = [x[1] for x in result_set[k]]
					# confs = [x[2] for x in result_set[k]]
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
						# selected_confs = [x for i, x in enumerate(confs) if i in selected]
						text_boxes = [[x, y] for x, y in zip(selected_texts, selected_boxes)]
						text_boxes = sorted(text_boxes, key=cmp_to_key(contour_sort))
						text_result = ""
						print(k)
						print(text_boxes)
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
						# conf_result = float(np.round(np.mean(selected_confs), 2))
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
						print("++++++++++++++entry in this block+++++++++++")
						texts = [x[0] for x in result_set[k]]
						bboxes = [x[1] for x in result_set[k]]
						for i, value in enumerate(zip(texts, bboxes)):
							print(list(value))
							if k not in list(final_result_set.keys()):
								final_result_set[k] = []
							final_result_set[k].append(list(value))
					else:
						if k not in list(final_result_set.keys()):
							final_result_set[k] = []
						final_result_set[k].append([result_set[k][0][0],result_set[k][0][1]])  
  
		print(final_result_set)
		# exit('++++++++++++++++++++')
		merge_surrounding(final_result_set, model_output)
		print("+++++++++++reached here after merge surrounding++++++++++")
		print(final_result_set)
        # exit('++++++++++++======')
		for k in list(final_result_set.keys()):
			all_values = final_result_set[k]
			print(all_values)
			# exit('________________')
			for value in all_values:
				draw.rectangle(value[1], outline=label2color[k], width=2)
				draw.text((value[1][0] + 5, value[1][1] - 20),
							text=k , fill=label2color[k], font=font)
		lookup_result = {}
		# t_page_end = datetime.now()
		# print("Time taken for page" + str(count) + ":", end=" ")
		# print(t_page_end - t_page_start)
		# print()
		# all_page_result["Page Number " + str(count)] = final_result_set
		# print(all_page_result)
		for k in list(final_result_set.keys()):
			if k in ["applicant_country", "beneficiary_country"]:
				for val in final_result_set[k]:
					result_country = lookup(val[0], 4, 90, "countries.txt", result_set, k)
					result_company = lookup(val[0], 4, 90, "organization.txt", result_set, k)
					# print(val)
					# print(result)
					# replacing ocr result with correct result
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
		# print(lookup_result)
		with open(os.path.join(result_path, file+ "_lookup.txt"), "w") as f:
			json.dump(lookup_result, f)
		print(f'final set : ++++++++++++++++++++++++++++++++++++++++++++')
		print(final_result_set)
		print(file)
		# exit('++++++++++++++++')
		with open(os.path.join(result_path, file + ".txt"), "w") as f:
			json.dump(final_result_set, f)
		new_img.save(os.path.join(result_path, file + ".png"))
