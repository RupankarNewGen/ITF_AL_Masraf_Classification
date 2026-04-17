import glob
import os
import json
from typing import List

import pandas as pd
import numpy as np


def area(coordinates):
	l = coordinates[2] - coordinates[0] + 1
	h = coordinates[3] - coordinates[1] + 1
	return l * h


# finds the minimum distance between a given bounding box
def minimum_distance(bb1, bb2):
	# bb1 points
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]
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


# finds and merges the modeloutput for text result, OCR confidence and Model confidence
def model_output_sum(key, box, data):
	# print(key, model_output[key],box)
	all_values = data[key]
	text_result = ""
	for value in all_values:
		# print(type(value[1]))
		try:
			inter_percent = get_intersection_percentage(box, value[1])
		# print("inter percent is!", inter_percent)
		except Exception as e:
			print(str(e))
			continue
		if inter_percent > 0.4:
			if text_result == "":
				text_result += value[0]
			else:
				text_result += " " + value[0]
	return text_result


# merge surrounding boxes
# def merge_surrounding(data):
# 	# new = data.copy()
# 	for key in list(data.keys()):
# 		# print(key)
# 		bboxes = [x[1] for x in data[key]]
# 		eps = 100
# 		all_values = data[key]
# 		length = len(all_values)
# 		if length > 1:
# 			i = 0
# 			while i in range(length - 1):
# 				# print(i)
# 				bb1 = all_values[i][1]
# 				bb2 = all_values[i + 1][1]
# 				# ocr_confs = [all_values[i][3],all_values[i+1][3]]
# 				min_dist = minimum_distance(bb1, bb2)
# 				try:
# 					IOU = get_iou_new(bb1, bb2)
# 					inter_percentage = get_intersection_percentage(bb1, bb2)
# 				except:
# 					i = i + 1
# 					continue
# 				if min_dist <= eps or IOU > 0 or inter_percentage > 0:
# 					# print("merging: " + all_values[i][0] + " and " + all_values[i+1][0])
# 					x_left = min(bb1[0], bb2[0])
# 					y_top = min(bb1[1], bb2[1])
# 					x_right = max(bb1[2], bb2[2])
# 					y_bottom = max(bb1[3], bb2[3])
# 					box = [x_left, y_top, x_right, y_bottom]
# 					text = model_output_sum(key, box, data)
# 					# print(text)
# 					# print("merged text is ", text)
# 					new_value = [text, box]
# 					# print(new_value)
# 					all_values.remove(all_values[i])
# 					all_values.remove(all_values[i])
# 					all_values.insert(i, new_value)
# 					# print(all_values)
# 					length = len(all_values)
# 					if length == 1:
# 						# print("will break")
# 						break
# 				else:
# 					# print("distance is very high")
# 					i = i + 1
# 		else:
# 			# print("will continue")
# 			continue

import numpy as np

# def minimum_distance_horizontal(bb1, bb2):
#     # Calculate the minimum horizontal distance between two bounding boxes
#     print('Horizental')
#     print(f'bounding box 1: {bb1} and bounding box 2: {bb2}')
#     x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
#     x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

#     min_distance_x = min(x2_bb1 - x1_bb2)
#     # min_distance_y = min(abs(y1_bb1 - y1_bb2), abs(y2_bb1 - y2_bb2))
#     print(min_distance_x)
#     # print(min_distance_y)

#     return min_distance_x
def minimum_distance_horizontal(bb1, bb2):
    # Calculate the minimum horizontal distance between two bounding boxes
    x1_bb1, _, x2_bb1, _ = bb1
    x1_bb2, _, x2_bb2, _ = bb2
    print('Horizental')
    print(f'bounding box 1: {bb1} and bounding box 2: {bb2}')

    min_distance_x = max(0, max(x1_bb1 - x2_bb2, x1_bb2 - x2_bb1))

    return min_distance_x

# def minimum_distance_horizontal(bb1, bb2):
#     # Calculate the minimum horizontal distance between two bounding boxes
#     x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
#     x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

#     min_distance_x = min(abs(x1_bb2 - x2_bb1), abs(x1_bb1 - x2_bb2))
#     min_distance_y = min(abs(y1_bb1 - y1_bb2), abs(y2_bb1 - y2_bb2))

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

def vertical_sort(tokens):
    # Sort tokens based on their y-coordinate value
    sorted_tokens = sorted(tokens, key=lambda token: token[1][1])

    return [sorted_tokens]  # Merge all tokens into a single group



def merge_surrounding(data):
	for key in list(data.keys()):
		bboxes = [x[1] for x in data[key]]
		eps_horizontal = 100  # Threshold for horizontal merging
		eps_vertical = 100
		print(key)    # Threshold for vertical merging
		print(data[key])
		print(len(data[key]))
		sorted_tokens = vertical_sort(data[key])
		data[key]=sorted_tokens[0]
		all_values = data[key]
		length = len(all_values)
		print(f'length of tokens: {length}')
		if length > 1:
			print(all_values)
			# exit('+++++++++++++')
			i = 0
			while i in range(length - 1):
				bb1 = all_values[i][1]
				bb2 = all_values[i + 1][1]
				min_dist_horizontal = minimum_distance(bb1, bb2)
				min_dist_vertical = minimum_distance_vertical(bb1, bb2)

				
				print(f'iteration: {i}')
				print(f'minimum distance vertical : {min_dist_vertical}')
				print(f'minimum distance horizental : {min_dist_horizontal}')
				print(f'bounding box 1: {bb1} and bounding box 2: {bb2}')
				# exit('++++++++++++')
				try:
					IOU_horizontal = get_iou_horizontal(bb1, bb2)
					IOU_vertical = get_iou_vertical(bb1, bb2)
					inter_percentage = get_intersection_percentage(bb1, bb2)
					# print(key)
					# print(f'minimum distance vertical : {min_dist_vertical}')
					# print(f'minimum distance horizental : {min_dist_horizontal}')
					# print(f'bounding box 1: {bb1} and bounding box 2: {bb2}')
				except:
					i = i + 1
					continue
				print(key)
				print(f'intersection percentage : {inter_percentage}')
				print(f'horizental intersection : {IOU_vertical}')
				print(f'vertical intersection : {IOU_horizontal}')
				if inter_percentage!=1.0:
					if (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) or \
						(min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage > 0):
						x_left = min(bb1[0], bb2[0])
						y_top = min(bb1[1], bb2[1])
						x_right = max(bb1[2], bb2[2])
						y_bottom = max(bb1[3], bb2[3])
						box = [x_left, y_top, x_right, y_bottom]
						text = model_output_sum(key, box, data)
						new_value = [text, box]
						all_values.remove(all_values[i])
						all_values.remove(all_values[i])
						all_values.insert(i, new_value)
						print(f'new value has added: {all_values}')
						length = len(all_values)
						if length == 1:
							break
					else:
						i = i+1
				else:
					all_values.remove(all_values[i+1])
					print(f'after remove: {all_values}')
					i=0
					if len(all_values)==1:
						break
			else:
				continue


	# finds intersection of two bounding boxes


# def get_intersection_percentage(bb1, bb2):
#     """
#     Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
#     """
#     # print('entered into gip function ++++++++++++++====')
#     # print(bb1['label'])
#     # print(bb2['word'])
#     # print(bb1['x1'])
#     # print(bb1['x2'])
#     # exit('')

#     assert bb1['x1'] < bb1['x2']
#     assert bb1['y1'] < bb1['y2']
#     assert bb2['x1'] < bb2['x2']
#     assert bb2['y1'] < bb2['y2']


#     # determine the coordinates of the intersection rectangle
#     x_left = max(bb1['x1'], bb2['x1'])
#     y_top = max(bb1['y1'], bb2['y1'])
#     x_right = min(bb1['x2'], bb2['x2'])
#     y_bottom = min(bb1['y2'], bb2['y2'])

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0

#     # The intersection of two axis-aligned bounding boxes is always an
#     # axis-aligned bounding box
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     # compute the area of both AABBs
#     bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
#     bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
#     #min_area = min(bb1_area,bb2_area)
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     if bb1_area>bb2_area:
#         intersection_percent = intersection_area / bb2_area
#     else:
#         intersection_percent = intersection_area / bb1_area
#         if intersection_percent<0.5:
#             intersection_percent=1               # if ocr bounding box is big then we need to consider the entire token if the intersection is less than o.5 also 
            

#     # print("The intersection percentage  of {text}, and {label},= {inter}".format(text = bb1['label'], label = bb2['word'], inter= intersection_percent))
#     # exit('++++++++++++++==')
#     assert intersection_percent >= 0.0
#     assert intersection_percent <= 1.0
#     return intersection_percent


# finds intersection over union of two bounding boxes
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
	assert bb1[0] <= bb1[2]
	assert bb1[1] <= bb1[3]
	assert bb2[0] <= bb2[2]
	assert bb2[1] <= bb2[3]

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

	# compute the area of both AABBs
	bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
	bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou


if __name__ == '__main__':
	folder_path: str = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/CS_complete_data/new_data_prep/ROOT_CS"

	#Master data path

	#Master data path

	images_path= os.path.join(folder_path, "Images")

	master_data_path= os.path.join(folder_path, "Master_Labels")

	# Main function
	data_path = os.path.join(folder_path, "New_Master_Data_Merged")


	if not os.path.exists(data_path):
		os.mkdir(data_path)

	# png file name
	# png_file_names: List = glob.glob(master_data_path + "/*.png")
	image_files = os.listdir(images_path)
	image_files = [x.split("_linking.png")[0] for x in image_files]

	print(image_files)
	# exit('+++++++++++=')

	for file in image_files:

		with open(os.path.join(master_data_path, file+"_labels.txt"), "r") as f:
			labels = json.load(f)
			print(file)
			merge_surrounding(labels)
			# exit('+++++++++=')
			print('yes')
			# print(f'''{file.split(".png")[0]}_labels.txt''')
			# exit('++++++++=')
			# print(labels)
		with open(os.path.join(data_path, file+'_labels.txt'), "w") as f:
			json.dump(labels, f)
		# exit('+++++++')