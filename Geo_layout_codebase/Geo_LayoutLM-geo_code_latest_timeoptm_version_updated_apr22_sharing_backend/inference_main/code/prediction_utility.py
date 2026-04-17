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
from constants import arial_file
from fuzzywuzzy import fuzz

App_Filepath = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config.read(App_Filepath + '/config.ini')

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
from constants import vertical_merge_labels, single_text_labels, master_keys
transform2 = transforms.ToPILImage()
transform = transforms.ToTensor()

result_path = config['PATH']['GEO_DUMP']


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
            print("prev_bbox >>>>>>>>>>>", prev_bbox)
            for single_bbox in bbox_data:
                print('single_bbox>>>>>>>>>>>>>', single_bbox)
                if are_on_same_line(prev_bbox, single_bbox, tolerance=line_tolerance):
                    if single_bbox not in all_bboxes_:
                        line_wise_index[idx].append(single_bbox)
                        all_bboxes_.append(single_bbox)
            line_wise_index[idx].sort(key=lambda x: x[0])
            initial_bbox.append(line_wise_index[idx][0])
            idx += 1
    print(initial_bbox)
    initial_bbox.sort(key=lambda x: x[1])
    print(initial_bbox)
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

def minimum_distance_vertical_old(bb1, bb2):
    # Calculate the minimum vertical distance between two bounding boxes
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    min_distance_y = min(abs(y1_bb2 - y2_bb1), abs(y1_bb1 - y2_bb2))

    return min_distance_y


def minimum_distance_vertical(bb1, bb2):
	# Calculate the minimum vertical distance between two bounding boxes
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2
	'''
	top to middle
	X-coordinate: x2
	Y-coordinate: ((y1_bb1 + y2_bb1) / 2 + y1_bb1) / 2 
	bottom to middle
	X-coordinate: x2
	Y-coordinate: ((y1_bb1 + y2_bb1) / 2 + y2_bb1) / 2
	'''
	min_distance_y = min(abs(y1_bb2 - y2_bb1), abs(y1_bb1 - y2_bb2), abs(y1_bb1 - y1_bb2), abs(y2_bb1 - y2_bb2), abs(y1_bb2 - (((y1_bb1 + y2_bb1) / 2 + y1_bb1) / 2)), abs(y2_bb2 - (((y1_bb1 + y2_bb1) / 2 + y2_bb1) / 2)))

	return min_distance_y


def check_vertical_indetween(bb1, bb2):
	in_between_flag = False
	# Calculate the minimum vertical distance between two bounding boxes
	x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
	x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2
	
	'''
	X-coordinate: x2
	Y-coordinate: ((y1 + y2) / 2 + y1) / 2
	'''
 
	if (y1_bb2 < y1_bb1 < y2_bb2) or (y1_bb2 < y2_bb1 > y2_bb2) or (y1_bb1 < y1_bb2 < y2_bb1) or (y1_bb1 < y2_bb2 < y2_bb1):
		in_between_flag = True
	return in_between_flag



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

def calculate_orientation(word_bbox):
	vertical_alignment_bbox = False
	# Calculate centroids of characters
	centroids = [(bbox[0] + bbox[2]) / 2 for bbox in word_bbox]

	# Calculate angle between horizontal axis and line connecting first and last centroids
	delta_y = centroids[-1] - centroids[0]
	delta_x = word_bbox[-1][2] - word_bbox[0][0]
	angle = np.arctan2(delta_y, delta_x)

	# Convert angle from radians to degrees
	angle_degrees = np.degrees(angle)
	print(angle_degrees)
	if angle_degrees < -50:
		vertical_alignment_bbox = True
	return vertical_alignment_bbox


def minimum_distance_old(bb1, bb2):
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



def minimum_distance(bb1, bb2):
	# bb1 points
	min_distance = 9999999999
	p_11 = np.array((bb1[0], bb1[1]))
	#(x2, (y1 + y2) / 2)
	# p_12 = np.array((bb1[0], bb1[3]))
	r_m_point = np.array((bb1[1], (bb1[3]+bb1[1])/2)) #???? r_m_point = np.array((bb1[2], (bb1[1] + bb1[3]) / 2))

	p_13 = np.array((bb1[2], bb1[3]))
	p_14 = np.array((bb1[2], bb1[1]))
	# all_points_bb1 = [p_11, p_12, p_13, p_14]
	all_points_bb1 = [p_11, p_13, p_14]#, r_m_point]
	# bb2 points
	p_21 = np.array((bb2[0], bb2[1]))
	p_22 = np.array((bb2[0], bb2[3]))
	p_23 = np.array((bb2[2], bb2[3]))
	# p_24 = np.array((bb2[2], bb2[1]))
	# all_points_bb2 = [p_21, p_22, p_23, p_24]
	all_points_bb2 = [p_21, p_22, p_23]
	for point1 in all_points_bb1:
		for point2 in all_points_bb2:
			dist = abs(np.linalg.norm(point1 - point2))
			if dist < min_distance:
				min_distance = dist
	return min_distance

def special_chr_check(bb_token, flag):
	if ',' in bb_token:
		flag = False
	if '-' in bb_token:
		flag = False
	return flag

def check_vertical_distribution(bb1, bb2):
	y1 = bb1[1]
	y2 = bb2[1]
	return abs(y1 - y2)


def area(coordinates):
	l = coordinates[2] - coordinates[0]
	h = coordinates[3] - coordinates[1]
	return l * h

def merge_surrounding_old(data, model_output):
	new = data.copy()
	print('entered into merging_surroundings ++++++++++++++++++++++++++++++++++++++')
	for key in list(data.keys()):
		print(key)
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
				# else:
				# 	print(f'length of all values: {len(all_values)}')
				# 	try:
				# 		bb1_area= calculate_bounding_box_area(bb1)
				# 		bb2_area= calculate_bounding_box_area(bb2)
				# 		if bb1_area>bb2_area:
				# 			all_values.remove(all_values[i+1])
				# 		else:
				# 			all_values.remove(all_values[i])
				# 	except Exception as e: 
				# 		pass
				# 	print("distance is very high")
				# 	i = i + 1
		else:
			print("will continue")
			continue

def after_skipping(all_values, inx_ele_skip): 
    # Add addition_val to the elements specified by inx_ele_skip
    # for index in range(len(inx_ele_skip)):
    #     inx_ele_skip[index] += addition_val 
    # Delete elements specified by inx_ele_skip 
    final_all_values = {}
    for idx, value in all_values.items():
        if idx not in inx_ele_skip:
            final_all_values[idx] = value
    # all_values = [value for idx, value in enumerate(all_values) if idx not in inx_ele_skip]
    return final_all_values

import copy
def merge_by_skipping_running(model_output, w, h, key, all_values):
	print("entered final validation", all_values)
	'''
	[["500 , telangana , india", [161, 172, 308, 180], 67.32535079575597], =>1
	["perak .", [290, 222, 329, 232], 84.68], =>2
	["dusun kabupaten i pauh utara deli , kecamatan sumatera serdang 20374 , hamparan indonesia -", [65, 223, 285, 270], 91.24132404181185]],
	                                          => 3 (need to merge 2 and 3)
	'''
	all_values = sorted(all_values, key=lambda bbox: bbox[1][0])#, reverse=True)
	all_values = {idx: item for idx, item in enumerate(all_values)}
	print("before starting", all_values)
	no_of_ele_skip = 1
	inx_ele_skip = [0]
	current_itteration = 0
	initial_check = True
	inx_ele_skip_flag = True
	len_idx_ele_skip = 1
	while len(all_values)>2:# and no_of_ele_skip-1 < len(all_values)-2:
		print('Before', all_values)
		if inx_ele_skip_flag:
			copied_inx_ele_skip = copy.deepcopy(inx_ele_skip)
			inx_ele_skip_flag = False
		if initial_check:
			actual_length = len(all_values)
			initial_check = False
		deep_copied_all_values = copy.deepcopy(all_values)
		all_values = after_skipping(all_values, inx_ele_skip)
		print('after_skipping these indexes', inx_ele_skip)
		print(all_values)
		length = len(all_values)
		print('>>>>>>>>>>>#################>>>>>>>>>>>>>>>>>>>>>>>>')
		print('>>>>>>>>>>>#################>>>>>>>>>>>>>>>>>>>>>>>>')
		# all_values = data[key]
		# for sort_idx in range(0,4):
		bboxes = [x[1] for x in all_values.values()]
		vertical_alignment_bbox = calculate_orientation(bboxes)
		vertical_alignment_bbox = False
		print(key, ">>>>>>> vertical_alignment_bbox >>>>>>>>>", vertical_alignment_bbox)
		# eps_horizontal = 100  # Threshold for horizontal merging
		# eps_vertical = 50  # Threshold for vertical merging
		# eps_vertical2 = 100  # Threshold for vertical merging
		eps_horizontal = round(w*5.5/100)#100
		eps_vertical = round(h*2.5/100) #100
		eps_vertical2 = 100  # Threshold for vertical merging
		if length > 1:
			i = 0
			while i in range(length - 1):
				print(i)
				bb1 = list(all_values.values())[i][1]
				bb2 = list(all_values.values())[i + 1][1]
				bb1_token = list(all_values.values())[i][0]
				bb2_token = list(all_values.values())[i + 1][0]
				confs = [list(all_values.values())[i][2], list(all_values.values())[i + 1][2]]
				min_dist_horizontal = minimum_distance(bb1, bb2)
				min_dist_vertical = minimum_distance_vertical(bb1, bb2)
				vertical_flag = check_vertical_indetween(bb1, bb2)
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
				if key in master_keys or vertical_alignment_bbox:
					merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) or (
							min_dist_vertical <= eps_vertical2 or IOU_vertical > 0 or inter_percentage)
				else:
					merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) and (
							min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage)# or vertical_flag) 
				print('min_dist_vertical', min_dist_vertical)
				print("IOU_horizontal > 0 or inter_percentage", IOU_horizontal, inter_percentage)
				if merge_flag:
					print('entered into second if')
					print("merging: " + list(all_values.values())[i][0] + " and " + list(all_values.values())[i + 1][0])
					x_left = min(bb1[0], bb2[0])
					y_top = min(bb1[1], bb2[1])
					x_right = max(bb1[2], bb2[2])
					y_bottom = max(bb1[3], bb2[3])
					box = [x_left, y_top, x_right, y_bottom]
					text = model_output_sum(key, box, model_output)
					print("merged text is ", text)
					avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
					avg_confs = float(np.round(avg_confs, 6))
					new_value = [text, box, avg_confs]
					print(new_value)
					print(all_values)
					print(i)
					deletionkey = next((k for k, v in all_values.items() if v == list(all_values.values())[i + 1]), None)
					updation_key = next((k for k, v in all_values.items() if v == list(all_values.values())[i]), None)
					print('deletion_key >>>>>>>>>>',deletionkey)
					print('updation_key >>>>>>>',updation_key)
					del all_values[deletionkey]        # we are deleting the right value/ after value and updating the left/previous value
					all_values[updation_key] = new_value
     
					# all_values.remove(all_values[i])
					# # all_values.remove(all_values[i]) ############??????
					# all_values.insert(i, new_value)
     
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
     
			print('after', inx_ele_skip, '$$$$$$$', all_values)
			######################################################
			######################################################
			#adding the skipped elements
			for itter_idx in inx_ele_skip:
				all_values[itter_idx] = deep_copied_all_values[itter_idx]
			print('After', all_values)
   
			#### sorting the values
			all_values = dict(sorted(all_values.items()))
   
			# updating the index with 1
			# for idx_, itter_idx_up in enumerate(inx_ele_skip):
			if max(all_values.keys()) not in inx_ele_skip:
				idx_ = 0
				increment_value = 1
				while idx_ < len(inx_ele_skip):
					req_val = inx_ele_skip[idx_] + increment_value
					if req_val in all_values:
						inx_ele_skip[idx_] = req_val
						idx_ = idx_ + 1
						increment_value = 1
					else:
						increment_value += 1
				######### break
    
    
			if max(all_values.keys()) in inx_ele_skip:
				initial_check = True
				if actual_length > len(all_values):
					inx_ele_skip = copied_inx_ele_skip
					inx_ele_skip_flag = True
				else:
					len_idx_ele_skip +=1
					inx_ele_skip = sorted(all_values.keys())[:len_idx_ele_skip]
     
			if len_idx_ele_skip+1 == len(all_values):
				print('i need to break here')
				print(inx_ele_skip)
				print(len_idx_ele_skip+2)
				print(len(all_values))
				break
		else:
			break
		

		print(all_values)
	return [item for item in all_values.values()]

def merge_surrounding(data, model_output, w, h):
	# print('start>>>>>>>>>>>>')
	# print(data)
	# print(model_output)
	# print(w, h)
	new = data.copy()
	print('entered into merging_surroundings ++++++++++++++++++++++++++++++++++++++')
	for key in list(data.keys()):
		print(key)
		if key in vertical_merge_labels or key in master_keys:
			# for validate in range(0,2):
			all_values = data[key]
   
			for sort_idx in range(0,4): #uncomment this later ####################################################
				########################################################################################################
				########################################################################################################
				# sort_idx = 0
				print('STARTED FOR ####### ITTERATION',sort_idx)
				bboxes = [x[1] for x in all_values]
				vertical_alignment_bbox = calculate_orientation(bboxes)
				print('vertical_alignment_bbox -->', vertical_alignment_bbox)
				vertical_alignment_bbox = False
				all_values = sorted(all_values, key=lambda bbox: bbox[1][sort_idx])
				print(f"sorting the values on the base of {sort_idx} ##### >>>>> {all_values}")
				'''
				[["house - 400 021 ,", [1291, 501, 1366, 532], 71.82], ["245 , martamo house cama road , mumbai", [1068, 516, 1300, 532], 81.86662341004609]]
				'''
				if len(all_values) > 1:
					# all_values = vertical_horizontal_values(new_all_values)
					# if key=='drawee_address':
					#     print(all_values)
					#     exit("PPPPPPPPPPPP")
					bboxes = [x[1] for x in data[key]]
					# eps_horizontal = 90  # Threshold for horizontal merging
					# eps_vertical = 50  # Threshold for vertical merging
					eps_horizontal = round(w*5.5/100)#100
					eps_vertical = round(h*2.5/100) #100
					eps_vertical2 = 100  # Threshold for vertical merging
					######################
					# if w>h:
					#     eps_horizontal = round(h*17/100)#100
					#     eps_vertical = round(w*12/100) #100
					# else:
					#     eps_horizontal = round(h*12/100)#100
					#     eps_vertical = round(w*17/100) #100
					############################
					# all_values = data[key]
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
							print(f"TOKENS GOING CHECKING FOR MERGING >> {bb1_token} >>and>> {bb2_token}")
		
							confs = [all_values[i][2], all_values[i + 1][2]]
							min_dist_horizontal = minimum_distance(bb1, bb2)
							min_dist_vertical = minimum_distance_vertical(bb1, bb2)
							vertical_flag = check_vertical_indetween(bb1, bb2)
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
							if key in master_keys or vertical_alignment_bbox:
								merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) or (
										min_dist_vertical <= eps_vertical2 or IOU_vertical > 0 or inter_percentage)
							else:
								merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) and (
										min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage)# or vertical_flag) 
								print("Yes entered in the else block", merge_flag)
							print('min_dist_vertical', min_dist_vertical)
							print(f"eps_horizontal = {eps_horizontal}, eps_vertical = {eps_vertical}")
							print("IOU_horizontal > 0 or inter_percentage", IOU_horizontal, inter_percentage)
							print("IOU_vertical > 0 or inter_percentage or vertical_flag", IOU_vertical, inter_percentage, vertical_flag)
							if merge_flag:
								print('entered into second if')
								print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
								x_left = min(bb1[0], bb2[0])
								y_top = min(bb1[1], bb2[1])
								x_right = max(bb1[2], bb2[2])
								y_bottom = max(bb1[3], bb2[3])
								box = [x_left, y_top, x_right, y_bottom]
								text = model_output_sum(key, box, model_output)
								print("merged text is ", text)
								avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
								avg_confs = float(np.round(avg_confs, 6))
								new_value = [text, box, avg_confs]
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
			########################################################################################################
			########################################################################################################
			print('before merge_by_skipping_running', all_values)
			data[key] = merge_by_skipping_running(model_output, w, h, key, all_values)\
       
			print("################# all_values #####################", all_values)
			# data[key] = all_values
		else:
			print(f'Vertical merging not happening ++++++++++ {key} ++++++++++++++')
			print(key)
			bboxes = [x[1] for x in data[key]]
			if w > h:
				v_eps = 10#round(h * 1.5 / 100)  # 10round(number)
				h_eps = 50 #36 #round(w * 5 / 100)  # 36
			else:
				v_eps = 10#round(h * 1.1 / 100)  # 10round(number)
				h_eps = 50#36 #round(w * 5.8 / 100)  # 36
			all_values = data[key]
			print('all_values >>>>>>>>>>>', all_values)
			length = len(all_values)
			if length > 1:
				i = 0
				while i in range(length - 1):
					print(i)
					bb1 = all_values[i][1]
					bb2 = all_values[i + 1][1]
					confs = [all_values[i][2], all_values[i + 1][2]]
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
						# try:
						avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
      
						# except:
						# 	avg_confs = 99.99
						"""if "NA" in ocr_confs:
                            avg_ocr_confs = "NA"
                        else:
                            avg_ocr_confs = ( ocr_confs[0]* area(bb1) + ocr_confs[1]*area(bb2) )/(area(bb1) + area(bb2))"""
						avg_confs = float(np.round(avg_confs, 6))
						new_value = [text, box, avg_confs]
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



            
def find_ocr_text_in_bbox(target_bbox, ocr_data):
    """
    Find all OCR text entries that fall within a target bounding box.
    
    Args:
        target_bbox (list): [x1, y1, x2, y2] coordinates of target box
        ocr_data (dict): Dictionary of OCR data with bounding boxes
    
    Returns:
        str: Concatenated text of all matching OCR entries
    """
    matching_text = []
    
    def is_within_bbox(box1, box2):
        # Check if box1 is within or overlaps significantly with box2
        x1, y1, x2, y2 = box1
        tx1, ty1, tx2, ty2 = box2
        
        # Calculate overlap area
        x_left = max(x1, tx1)
        y_top = max(y1, ty1)
        x_right = min(x2, tx2)
        y_bottom = min(y2, ty2)
        
        if x_right > x_left and y_bottom > y_top:
            overlap_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (x2 - x1) * (y2 - y1)
            overlap_ratio = overlap_area / box1_area
            return overlap_ratio > 0.5
        return False

    # Find all OCR entries that fall within the target bbox
    for ocr_id, ocr_entry in ocr_data.items():
        ocr_bbox = ocr_entry['bbox']
        if is_within_bbox(ocr_bbox, target_bbox):
            matching_text.append(ocr_entry['text'])
    
    return ' '.join(matching_text)

def process_results_with_ocr(results, ocr_data):
	"""
	Process results dictionary and add OCR text for each bbox entry.

	Args:
		results (dict): Dictionary containing results with bboxes
		ocr_data (dict): Dictionary of OCR data

	Returns:
		dict: Updated results dictionary with OCR text
	"""
	processed_results = {}

	for key, value_list in results.items():
		processed_results[key] = []
		for entry in value_list:
			# Each entry is [text, bbox, confidence]
			text, bbox, confidence = entry
			ocr_text = find_ocr_text_in_bbox(bbox, ocr_data)
			ratio = fuzz.ratio(str(text).lower().replace(' ', ''), str(ocr_text).lower().replace(' ', ''))
			# Append the OCR text as the fourth element
			if ratio == 100:  # You can adjust this threshold
				# processed_results[key].append([text, bbox, confidence, ocr_text]) #1**
				processed_results[key].append([ocr_text, bbox, confidence])
			else:
				processed_results[key].append([text, bbox, confidence])
				
	return processed_results


#result_path = '/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/val_inference_files/dataset/results'
def result_generation(img_path, token_data, all_words_):
	# input_path= "/home/ntlpt19/Downloads/MERGED_DATA/GEO_Latest/geolayoutlm_code_base_2/CI_EVAL/val_inference_files/dataset/custom_trial"
	# file_path= os.path.join(input_path,'Results/annotations')
	# result_path = os.path.join(input_path, "Results_CS_validated")
	# if not os.path.exists(result_path):
	#         os.mkdir(result_path)
	# img_path= os.path.join(input_path, "Results/images")
	# image_list= os.listdir(file_path)
	# image_list=  [ image.split('_tagging.json')[0] for image in image_list if image.endswith(".json")]
	# # image_list= 
	# print(image_list)

	# Open the file in write mode and save the text
 
	# with open('token_data.txt', 'w') as file:
	# 	file.write(str(token_data))
	# exit('OKKKKKK')
 
	number_of_colors = 80
	color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
	# exit('+++++++++++++++++++')
	count=0
	# for file in image_list:
	file = os.path.splitext(os.path.basename(img_path))[0]
	count+=1
	im = Image.open(img_path)
	for i, image in enumerate(ImageSequence.Iterator(im)):
		w, h = image.size
		temp = image.convert("L")
		image_data = np.asarray(temp)
		image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
		
		# plt.imshow(image)
		# plt.show()
		# plt.close()
		arr = transform(image)
		
		# try:
		#     with open(result_json) as f:
		#         token_data = json.load(f)['form']
		# except Exception as e:
		#     with open(result_json) as f:
		#         token_data = json.load(f)
		predicted_keys = [token["pred_key"] for token in token_data if not token["pred_key"]=='O']
		labels= [token.split('-')[1]for token in predicted_keys]
		labels= set(labels)
		label2color = {}
		for i, l in enumerate(labels):
			label2color[l] = color[i]
		font = ImageFont.truetype(arial_file, 20)
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
					result_set[(token_data[i]['pred_key']).split('-')[1]].append([token_data[i]['text'], token_data[i]['coords'], float(np.round(token_data[i].get('confidence', 0.00), 6))])
		print(result_set)
  
		result_set = process_results_with_ocr(result_set, all_words_) # added
		model_output = result_set.copy()
  
		with open(os.path.join(result_path, file + str(count) + "model_output.txt"), "w") as f:
			json.dump(result_set, f)
		f.close()
		print(result_set)
		########################################### #2**
		# print(updated_result)
		# with open(os.path.join(result_path, file + str(count) + "model_output_updated.txt"), 'w') as frs_:
		# 	frs_.write(str(updated_result))
		# frs_.close()
		#################################
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
					try:
						confs = [x[2] for x in result_set[k]]
					except:
						confs = [0.00 for x in result_set[k]]
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
						selected_confs = [x for i, x in enumerate(confs) if i in selected]
						text_boxes = [[x, y] for x, y in zip(selected_texts, selected_boxes)]
						text_boxes = sorted(text_boxes, key=cmp_to_key(contour_sort))
						text_boxes = validate_contour_sort(text_boxes)
						text_result = ""
						print(k)
						print(text_boxes)
						for tb in text_boxes:
							if text_result == "":
								text_result += tb[0]
							else:
								text_result += " " + tb[0]
      
						# for i, tb in enumerate(text_boxes):
						# 	text_token = tb[0]
						# 	# Check if we need to add a space:
						# 	# - Only if the previous character in text_result is alphanumeric
						# 	# - Only if the current token doesn't start with a space
						# 	if i > 0 and text_result[-1].isalnum() and text_token[0].isalnum() and not text_token.startswith(" "):
						# 		text_result += " " + text_token
						# 	else:
						# 		text_result += text_token

      
						print(text_result)
						x1 = min([x[0] for x in selected_boxes])
						x2 = max([x[2] for x in selected_boxes])
						y1 = min([x[1] for x in selected_boxes])
						y2 = max([x[3] for x in selected_boxes])
						box_result = [x1, y1, x2, y2]
						conf_result = float(np.round(np.mean(selected_confs), 6))
						# print(box_result)
						if k not in list(final_result_set.keys()):
							final_result_set[k] = []
						final_result_set[k].append([text_result, box_result, conf_result])

				else:
					if k not in list(final_result_set.keys()):
						final_result_set[k] = []
					try:
						final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], result_set[k][0][2]])
					except:
						final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], 0.00])
         
			else:
				if len(result_set[k]) > 1:
					print("++++++++++++++entry in this block+++++++++++")
					texts = [x[0] for x in result_set[k]]
					bboxes = [x[1] for x in result_set[k]]
					try:
						confs = [float(np.round(x[2], 6)) for x in result_set[k]]
					except:
						confs = [0.00 for x in result_set[k]]
					for i, value in enumerate(zip(texts, bboxes, confs)):
						print(list(value))
						if k not in list(final_result_set.keys()):
							final_result_set[k] = []
						final_result_set[k].append(list(value))
				else:
					if k not in list(final_result_set.keys()):
						final_result_set[k] = []
					try:
						# final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], result_set[k][0][2]])
						final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], float(np.round(result_set[k][0][2], 6))])
					except:
						final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], 0.00])
					

		print(final_result_set)
		merge_surrounding(final_result_set, model_output, w, h)
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

		print(file)
		# exit('++++++++++++++++')
		with open(os.path.join(result_path, file + ".txt"), "w") as f:
			json.dump(final_result_set, f)
		new_img.save(os.path.join(result_path, file + ".png"))
		
	return final_result_set