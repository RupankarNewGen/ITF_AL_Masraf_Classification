import glob
import os
import json
import pandas as pd
from fuzzywuzzy import fuzz
import re

from post_processing_master_gt import generic_page_no
from post_processing_pred import pp_cash_drawn_rules

from utility import check_and_modify_labels, merge_top_bottom_keys


from typing import List, Dict


""" This script used to generate accuracy generation of the dataset and 
used documents like PL, CS, COO, BOL"""
def remove_spaces(s):
    # Removes spaces from a string
    return "".join(char for char in s if char != " ")

def remove_start_end_spl_char(actual_text: str, pred_text:str):
	print(f'actual text : {actual_text}')
	print(f'pred text : {pred_text}')

	actual_text= actual_text.split(' ')
	pred_text= pred_text.split(' ')
	print(actual_text)
	print(pred_text)
	if len(actual_text)== len(pred_text):
		for actual, pred in zip(actual_text, pred_text):
			print('length of actual and predicted is same+++++++++')
			print(actual)
			print(pred)
			
			actual_indics= actual_text.index(actual)
			print(actual_indics)
			pred_indices= pred_text.index(pred)
			print(pred_indices)
			if actual_indics==0:
				a_text= "".join(ch for ch in actual if ch.isalnum() or ch=='.')
				actual_text[actual_indics]= a_text
			if actual_indics==len(actual_text)-1:
				a_text= "".join(ch for ch in actual if ch.isalnum())
				actual_text[actual_indics]= a_text

			if pred_indices==0:
				p_text="".join(ch for ch in pred if ch.isalnum() or ch== '.')
				pred_text[pred_indices]= p_text
			if pred_indices== len(pred_text)-1:
				p_text="".join(ch for ch in pred if ch.isalnum())
				pred_text[pred_indices]= p_text

	else:
		for word in pred_text:
			word_indices=  pred_text.index(word)
			if word_indices==0 :
				a_text= "".join(ch for ch in word if ch.isalnum() or ch=='.')
				pred_text[word_indices]= a_text
			elif word_indices== len(pred_text)-1:
				a_text= "".join(ch for ch in word if ch.isalnum())
				pred_text[word_indices]= a_text
			else:
				pass
		for word in actual_text:
			word_indices=  actual_text.index(word)
			if word_indices==0 :
				a_text= "".join(ch for ch in word if ch.isalnum() or ch=='.')
				actual_text[word_indices]= a_text
			elif word_indices== len(actual_text)-1:
				a_text= "".join(ch for ch in word if ch.isalnum())
				actual_text[word_indices]= a_text
			else:
				pass
		

	actual_text= " ".join(actual_text)
	pred_text= " ".join(pred_text)
	return actual_text,pred_text

def remove_spl_char_multiple_pred(text: str):
	text= text.split(' ')
	for word in text:
		word_indices= text.index(word)
		if word_indices==0 or word_indices==len(text)-1:
			a_text= "".join(ch for ch in word if ch.isalnum())
			text[word_indices]= a_text
	actual_text= " ".join(text)
	print(f'actual text: {actual_text}')
	return actual_text
	

def fuzzy_compare_ignore_spaces(str1, str2):
    # Remove spaces from both strings
    str1_without_spaces = remove_spaces(str1)
    str2_without_spaces = remove_spaces(str2)
    print(str1_without_spaces)
    print(str2_without_spaces)

    # Calculate the Jaccard index between the modified strings
    similarity = fuzz.token_set_ratio(str1_without_spaces, str2_without_spaces)

    # You can adjust the threshold based on your requirement
    threshold = 80  # For example, consider strings with similarity 80 or higher as similar

    return similarity >= threshold


def remove_alphabets(text):
	text = re.sub(r'[a-zA-Z]', '', text)
	return text

def preprocess_incoterm(text : str, incoterm_list: List):
  term = text[0]
  text =text.split(' ')
  for word in text:
    if word.upper() in incoterm_list:
      term= word
    else:
      pass
    if term  is not None:
      return term

	






# def remove_spl_char(actual_text: str, pred_text:str):
#     print(f'actual text : {actual_text}')
#     print(f'pred text : {pred_text}')

#     actual_text= actual_text.split('')
#     pred_text= pred_text.split('')

#     for actual, pred in (actual_text, pred_text):
#         actual_indics= actual_text.index(actual)
#         pred_indices= pred_text.index(pred)

#         a_text="".join(ch for ch in actual if ch.isalnum())
#         p_text="".join(ch for ch in pred if ch.isalnum())

#         actual_text[actual_indics]= a_text
#         pred_text[pred_indices]= p_text


        
        


def append_values(new_row, actual_value, predicted, to_do):
	print("value1 is", actual_value)
	print("list of predicted values are", predicted)
	print("to_do is", to_do)
	l2 = len(predicted)
	for j in range(l2):
		# print(" j is", str(j))
		intersection = get_iou_new(actual_value[1], predicted[j][1])
		if intersection > 0.25:
			# print("match found")
			print(f'entered after intersection {actual[i][0]}')
			exit()
			new_row.append(actual[i][0])
			new_row.append(predicted_label[j][0])
			accuracy = fuzz.ratio(str(new_row[2]).lower(), str(new_row[3]).lower())
			new_row.append(accuracy)
			if accuracy == 100:
				new_row.append(1)
			else:
				new_row.append(0)
			new_row.append(predicted_label[j][2])
			new_row.append(actual_value[1])
			data.append(new_row)
			try:
				to_do.remove(j)
			except:
				pass
			# print("going to  break")
			break
	else:
		# print("else executed")
		# print(new_row)
		new_row.append(actual_value[0])
		new_row.append("")
		new_row.append(0)
		new_row.append(0)
		new_row.append(0)
		new_row.append(actual_value[1])
		data.append(new_row)
	# print("new_to_do is",to_do)
	return to_do





def filter_prediction(new_row, actual_value, predicted, to_do):
	l2 = len(predicted)
	print(predicted)
	print(f'The no of unique predictions: {l2}')
	print(f'The actual value insider filter: {actual_value[0].lower(), type(actual_value[0].lower())}')
	print(f'the prediction value: {predicted[0][0].lower(), type(predicted[0][0].lower())}')
	print(f' to-do value: {to_do}')
	# accuracy = fuzz.ratio(actual_value[0].lower(), predicted[0][0].lower())
	# print(accuracy)
	flag= True
	for j in range(l2):
		print('Entered into for loop ++++++++++++++++++++++++++++++++')
		actual= actual_value[0].lower()
		# actual= actual.strip()
		actual= actual.replace(' ','')
		pred= predicted[j][0].lower()
		# predicted= predicted.strip()
		pred= pred.replace(' ','')

		print(f'The actual value inside for loop : {actual_value[0].lower(), type(actual_value[0].lower())}')
		print(f'the prediction value inside for loop: {predicted[0][0].lower(), type(predicted[0][0].lower())}')
		if (actual.lower())== (pred.lower()) and flag==True:
			print('entered into filter +++++++++++++++++===')
			flag= False                                   # consider one prediction 
			new_row.append(actual_value[0])
			new_row.append(predicted[j][0])
			accuracy = fuzz.ratio(str(actual).lower(), str(pred).lower())
			new_row.append(accuracy)
			if accuracy == 100:
				new_row.append(1)
			# new_row.append(predicted[j][2])
			new_row.append(actual_value[1])
			print(f'Row Inserted: {new_row}')
			data.append(new_row)
			print(data)
			try:
				to_do.remove(j)
			except:
				pass
			# print("going to  break")
			break
	else:
		print("else executed")
		# exit('+++++++++++++')
		# print(new_row)
		new_row.append(actual_value[0])
		new_row.append("")
		new_row.append(0)
		new_row.append(0)
		new_row.append(0)
		new_row.append(actual_value[1])
		data.append(new_row)
	# print("new_to_do is",to_do)
	return to_do



# get intersection over union of two bounding boxes
def get_iou_new(bb1, bb2):
	try:
		assert bb1[0] < bb1[2]
		assert bb1[1] < bb1[3]
		assert bb2[0] < bb2[2]
		assert bb2[1] < bb2[3]
	except:
		return 0

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


def filter_address(x):
	x = str(x)
	x = x.strip()
	new_x = re.sub(r'\s+', ' ', x)
	print("newx")
	print(new_x)
	# remove the special characters like comma, semicolon etc
	new_x = "".join([x for x in new_x if x.isalnum() or x in [" "]])
	print(new_x)
	return new_x


def fuzzy_float_comparison(float1, float2, tolerance=1e-6):
	absolute_difference = abs(float1 - float2)
	print(f'absolute difference: {absolute_difference}')
	if absolute_difference <= tolerance:
		return 100  # Return 100 for a perfect match within the tolerance
	else:
		# Calculate a similarity score based on the relative difference
		similarity = 100 - (absolute_difference / max(abs(float1), abs(float2))) * 100
		return similarity

def remove_keys_from_dict(input_dict, keys_to_remove):
    # Use dictionary comprehension to create a new dictionary without the specified keys
    return {key: input_dict[key] for key in input_dict if key not in keys_to_remove}

if __name__ == '__main__':
	# count = 0
	occurrences = {}
	data = []

	folder_path: str = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/CS_complete_data/new_data_prep/ROOT_CS"
	# folder_path: str = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/Layoutlmv3_performance_phase_1_data/CS_jul_13/evaluation_data"

	result_path = os.path.join(folder_path, "Results_CS_validated")
	data_path = os.path.join(folder_path, "New_Master_Data_Merged")

	# import json
	data_files = os.listdir(data_path)
	result_files = os.listdir(result_path)
	incoterm_list=[]

	# with open("incoterm_list.txt") as fp:
	# 	for line in fp:
	# 		incoterm_list.append(line.strip())
	print(incoterm_list)
	# exit('++++++++++')

	# check how many pngs are there in those two folders
	print("Number of images in results", len(glob.glob(result_path + "/*.png")))
	print("Number of images in master data", len(glob.glob(data_path + "/*_labels.txt")))

	# exit("+++++++++++")

	# validating the number of files coming from both these folders
	print(f"number of files coming from the Master data folder: {len(data_files)}")
	print(f"number of files coming from the Result cs validated folder: {len(result_files)}")

	# exit("++++++++++++++")

	new = []
	new_result = []

	# FINDING LIST OF DATAFILES AND RESULT FILES
	num_labels_files_traversed: int = 0
	for file in data_files:
		if str(file)[-10:] == "labels.txt":
			print(f"file number is {num_labels_files_traversed}")
			num_labels_files_traversed = num_labels_files_traversed + 1
			# print(str(file)[-10:])
			new.append(file)
			new_result.append(file[:-11] + ".txt")

	# exit("++++++++++++++++")
	data_files = new
	result_files = new_result
	# print(result_files)
	print(len(data_files))
	print(len(result_files))
	# exit("+++++++++")

	# creating a dictionary containing number of occurrences of all our fields in our dataset.
	for count, file in enumerate(data_files):
		print("count is:", count)
		with open(os.path.join(data_path, file), "r") as f:
			labels = json.load(f)
			try:
				labels = list(labels)
			except:
				continue
		for label in labels:
			if label in occurrences:
				occurrences[label] += 1
			else:
				occurrences[label] = 1
	column_names = []
	names = list(occurrences.keys())

	print(f'Number of classes used : {len(names)}')
	# print(names)
	# exit("+++++++++++++++++")
	column_names.append("File_Name")
	column_names.append("label_name")
	column_names.append("actual")
	column_names.append("predicted")
	column_names.append("Accuracy")
	column_names.append("Match/No_Match")
	column_names.append("model_confidence")
	column_names.append("bbox")
	print(column_names)
	print(len(column_names))
	# name = str(result_files[i])[0:-4]
	for file, predicted_files in zip(data_files, result_files):
		print("file name is:", file)
		print("resulted filename is", predicted_files)
		# exit('++++++++++++++=')
		# continue
		# finding number of characters and type of document.
		with open(os.path.join(data_path, file), "r") as f:
			labels = json.load(f)
		try:
			print(os.path.join(result_path, file[0:-11] + ".txt"))
			# exit('+++++++++++++++==')
			with open(os.path.join(result_path, file[0:-11] + ".txt"), "r") as f2:
				predicted = json.load(f2)
		except:
			# print("some problem opening file")
			try:
				print(f'''printing the path: {result_path, file[0:-11] + ".txt"}''')
				with open(os.path.join(result_path, file[0:-11] + ".txt"), "r") as f2:
					predicted = json.load(f2)
			# print("opened")
			except:
				print("still not opened")
				continue

		print(f'acutal labels: {labels}')
		print(f'number of keys : {len(list(labels.keys()))}')
		print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++=')
		print(f'predicted labels: {predicted}')
		print(f'predicted labels: {len(predicted)}')
		# exit('++++++++++++++++++++++++')
		# Removing unnecessary labels from the classes.txt file
		keys_to_remove = ['currency_amount','signed_stamp','drawer_country','drawee_country'
		,'drawee_bank_country','doc_settlement_instructions','document_enclosed']
		#['currency_amount','doc_settlement_instructions','document_enclosed','signed_stamp']
		labels = remove_keys_from_dict(labels, keys_to_remove)
		predicted = remove_keys_from_dict(predicted, keys_to_remove)



		# # Merging top bottom addresses

		# key_name_changes = {
		# 	"drawer_bank_bottom_address": "drawer_bank_address",
		# 	"drawer_bank_bottom_name": "drawer_bank_name",
		# 	"drawer_bank_bottom_bic" : "drawer_bank_bic"
        #                  }
		# labels= merge_top_bottom_keys(key_name_changes, labels)

		# predicted= merge_top_bottom_keys(key_name_changes, predicted)

		# exit('++++++++++++++===')
		top_value = ["drawer_bank_address",'drawer_bank_name','drawer_bank_bic']
		bottom_value = ["drawer_bank_bottom_address",'drawer_bank_bottom_name','drawer_bank_bottom_bic']
		if len(top_value)== len(bottom_value):
			for i in range(len(top_value)):
				labels = check_and_modify_labels(labels,bottom_value[i], top_value[i])
				predicted = check_and_modify_labels(predicted,bottom_value[i], top_value[i])
		else:
			print('The len of top and bottom values should be same')
				
		# exit('+++++++++++==')

		if predicted == {} and labels == {}:
			print("*******")
			print(file)
			continue

		for key in list(occurrences.keys()):
			# print("key is", key)
			# if key not in ['currency_amount''doc_settlement_instructions''document_enclosed''signed_stamp']
			row = []
			row.append(file[0:-11] + ".png")
			if key in labels and key in predicted:
				if len(labels[key]) == 1 and len(predicted[key]) == 1:
					row.append(key)
					row.append(str(labels[key][0][0]))
					row.append(str(predicted[key][0][0]))
					# print(f'row data: {row}')
					# print(str(row[2]).lower(), str(row[3]).lower())
					#remove special chars in starting and ending of the string
					if key in['drawer_name']:

						print(f'the row[2] value: {row[2]}')
						print(f'the row[3] value: {row[3]}')
					# exit('++++++++++++==')
					if key not in ['csh_bill_amount','csh_presentation_date','page_no']:
						row[2], row[3] = remove_start_end_spl_char(row[2], row[3])
						print(f'after remove spl chars: {row}')
					# exit()
					# post-processing for 

					if key in ['net_weight', 'gross_weight','total_quantity_of_goods']:
						print('entered into  remove alphabets++++++++++++++++')
						print(row[2])
						print(row[3])
						row[2] = remove_alphabets(row[2])
						row[3] = remove_alphabets(row[3])
						print(row[2])
						print(row[3])
					# processing for page no
					if key in ["incoterm"]:
						row[2] = preprocess_incoterm(row[2], incoterm_list)
						row[3] = preprocess_incoterm(row[3], incoterm_list)
					
					# post-processing for address
					if key in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
							"drawee_address"]:
						row[2] = filter_address(row[2])
						row[3] = filter_address(row[3])
						print(f'The actual value: {row[2]}')
						print(f'The pred value: {row[3]}')
						# exit('+++++++++++===')

					# processing for page no
					if key in ["page_no"]:
						row[2] = generic_page_no(row[2])
						row[3] = generic_page_no(row[3])
						print(f'the row[2] value: {row[2]}')
						print(f'the row[3] value: {row[3]}')
						print(f'type of row[2]: {type(row[2])}')
						print(f'type of row[3]: {type(row[3])}')
						# exit('___________________')

					# processing for cash drawn under rules
					if key in ["csh_drawn_under_rules"]:
						row[2] = pp_cash_drawn_rules(row[2])
						row[3] = pp_cash_drawn_rules(row[3])
						print(f'The actual value: {row[2]}')
						print(f'The pred value: {row[3]}')
						# exit('+++++++++++++++++==')

					print(key)
					print(f'the row[2] value: {row[2]}')
					print(f'the row[3] value: {row[3]}')
					print(f'type of row[2]: {type(row[2])}')
					print(f'type of row[3]: {type(row[3])}')
					# accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
					# print(f'accuracy of {key}: {accuracy}')
					if key == 'gross_weight' or key== "net_weight" or key== "total_quantity_of_goods" or key=='csh_bill_amount':
						try:
							row_2= remove_spaces(row[2])
							row_3=  remove_spaces(row[3])
							accuracy = fuzz.ratio(float(str(row_2)), float(str(row_3)))
							print(f'type of row[2]: {type(int(row[2]))}')
							print(f'type of row[3]: {type(int(row[3]))}')
							print(f'accuracy of {key}: {accuracy}')
						except Exception as e:
							try:
								accuracy = fuzzy_float_comparison(float(str(row_2)), float(str(row_3)))
								print(f'Executed second accuracy {key}: {accuracy}')
							except Exception as e:
								print(f'the row[2] value: {row[2]}')
								print(f'the row[3] value: {row[3]}')
								accuracy = fuzz.ratio(str(row_2).lower(), str(row_2).lower())
								print(f'Executed third accuracy {key}: {accuracy}')
					if key in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
							"drawee_address","page_no", "doc_charge_instructions",
							"doc_delivery_instruction","csh_bill_currency","csh_presentation_date","csh_due_date",'tenor_indicator','tenor_indicator_date',
							'tenor_indicator_type','tenor_type','usance_tenor','drawee_country',
							'drawer_country','nostro_bank_country','drawer_bank_country',
							'drawee_bank_country','drawer_name','csh_presentation_date',
							'drawer_address','drawee_name','drawer_bank_name','drawee_bank_name','drawer_bank_bic']:
						print("&&&&&&&&&&&&&&&")
						if key in ['tenor_indicator_date','tenor_indicator_type','tenor_type','usance_tenor',
						'csh_bill_currency','drawer_name','drawee_name','csh_presentation_date','drawer_bank_country','drawee_bank_country','nostro_bank_country']:
							if fuzzy_compare_ignore_spaces(row[2], row[3]):
								accuracy = 100
							# else:
							# 	accuracy=0
						else:
							print('Entered+++++++++++++++++++++=')
							if row[2] and row[3] is not None:
								row_2= remove_spaces(row[2])
								row_3=  remove_spaces(row[3])
								accuracy = 100 if str(row_3).__contains__(row_2) else fuzz.ratio(str(row_2).lower(), str(row_3).lower())
								print("accuracy:", accuracy)
							else:
								accuracy=0
					else:
						if row[2] and row[3] is not None:
							row_2= remove_spaces(row[2])
							row_3=  remove_spaces(row[3])
							accuracy = 100 if str(row_3).__contains__(row_2) else fuzz.ratio(str(row_2).lower(), str(row_3).lower())
						else:
							accuracy=0

					# if key=="csh_drawn_under_rules":
					# 	print(f'accuracy: {accuracy}')
					# 	exit('++++++++++++==')
					print(f'accuracy: {accuracy}')
					row.append(accuracy)

					if accuracy == 100:
						row.append(1)
					else:
						row.append(0)
					# print(row)
					# row.append(predicted[key][0][2])
					row.append(labels[key][0][1])
					data.append(row)
					# print(data)
					# exit('+++++++')
				else:
					# exit()
					row.append(key)
					actual = labels[key]
					predicted_label = predicted[key]
					print(f'second condition: {actual}')
					print(f'second condition: {predicted_label}')
					print(row)
					l1 = len(actual)
					l2 = len(predicted_label)
					if key in ['net_weight', 'gross_weight','total_quantity_of_goods']:
						for i in range(l2):
							predicted_label[i][0]=remove_alphabets(predicted_label[i][0])
					if key in ["incoterm"]:
						for i in range(l2):
							print(f'incoterm predicted: {predicted_label[i][0]}')
							predicted_label[i][0]=preprocess_incoterm(predicted_label[i][0], incoterm_list)

					if key in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
							"drawee_address"]:
						print('address block ++++++++++++++++++++++')
						print(actual)
						print(predicted_label)
						# exit('++++++++++++')
						for i in range(l2):
							predicted_label[i][0] = filter_address(predicted_label[i][0])
						for i in  range(l1):
							actual[i][0] = filter_address(actual[i][0])
						print(actual)
						print(predicted_label)
						# exit('+++++++++++++++++++')
					if key in ["page_no"]:
						for i in range(l2):
							predicted_label[i][0] = generic_page_no(predicted_label[i][0])
						for i in  range(l1):
							actual[i][0] = generic_page_no(actual[i][0])

					# processing for cash drawn under rules
					if key in ["csh_drawn_under_rules"]:
						for i in range(l2):
							predicted_label[i][0] = pp_cash_drawn_rules(predicted_label[i][0])
						for i in  range(l1):
							actual[i][0] = pp_cash_drawn_rules(actual[i][0])
					for i in range(l2):
							print(row)
							print(predicted_label)
							print(f'The  predicted  value is: {predicted_label[i][0]}')
							# exit('+++++++++++++==')
							predicted_label[i][0]=remove_spl_char_multiple_pred(predicted_label[i][0])
					print(f'second condition after remove alphabets: {predicted_label}')
					to_do = [*range(0, l2, 1)]
					print("starting to_do are", to_do)
					for i in range(l1):
						new_row = row.copy()
						actual_value = actual[i]
						actual[i][0]=remove_spl_char_multiple_pred(actual_value[0])
						print(f'the actual value is: {actual_value}')
						# exit('+++++++==')

						to_do = filter_prediction(new_row, actual_value, predicted_label, to_do)
					print(to_do)
					# exit('+++++++++++++')
					# # print("remaining to_do are", to_do)
					# # print(predicted_label)
					# for i in to_do:
					# 	new_row = row.copy()
					# 	new_row.append("")
					# 	new_row.append(predicted_label[i][0])
					# 	new_row.append(0)
					# 	new_row.append(0)
					# 	new_row.append(predicted_label[i][2])
					# 	new_row.append(predicted_label[i][1])
					# 	data.append(new_row)
			elif key in labels and key not in predicted:
				# exit()
				print(f'key is : {key}')
				print('entered into third condition')
				print(row)
				# exit('+++++++++++++===')
				row.append(key)
				if len(labels[key]) == 1:
					new_row = row.copy()
					new_row.append(str(labels[key][0][0]))
					new_row.append("")
					new_row.append(0)
					new_row.append(0)
					new_row.append(0)
					new_row.append(labels[key][0][1])
					data.append(new_row)
				else:
					for val in labels[key]:
						new_row = row.copy()
						new_row.append(val[0])
						new_row.append("")
						new_row.append(0)
						new_row.append(0)
						new_row.append(0)
						new_row.append(val[1])
						data.append(new_row)
			# print(row)
			elif key not in labels and key in predicted:
				# exit()
				row.append(key)
				row.append("")
				if len(predicted[key]) == 1:
					row.append(str(predicted[key][0][0]))
					row.append(0)
					row.append(0)
					# row.append(predicted[key][0][2])
					row.append(predicted[key][0][1])
					# data.append(row)
				else:
					for val in predicted[key]:
						new_row = row.copy()
						new_row.append(val[0])
						new_row.append(0)
						new_row.append(0)
						new_row.append(val[2])
						new_row.append(val[1])
						data.append(new_row)
			else:
				continue
	print(row)
	# exit('++++++++++++++++++')
	df = pd.DataFrame(data, columns=column_names)
	# pd.set_option('display.max_columns', None)
	# exit('+++++++++++==')

	print(df["Match/No_Match"].sum())
	print(df["Match/No_Match"])
	# exit()
	# df.loc[((df["label_name"].isin(['drawer_bank_address',
    # 'drawer_name',
    # 'drawer_address',
    # 'drawee_name',
    # 'drawee_address',
    # 'drawer_bank_name',
    # 'drawer_bank_adress',
	# 'nostro_bank_name',
	# 'nostro_bank_address',
	# 'drawee_bank_name',
	# 'nostro_bank_bic','doc_delivery_instruction','csh_due_date'])) & (df["Accuracy"].apply(int) >=75)), "Match/No_Match"] = 1


	df.loc[((df["label_name"].isin(['drawer_address','nostro_bank_bic',
	'doc_delivery_instruction','drawee_name'])) & (df["Accuracy"].apply(int) >=80)), "Match/No_Match"] = 1


	# df.loc[((df["label_name"].isin(['doc_charge_instructions'])) & (df["Accuracy"].apply(int) >=60))|(df["model_confidence"].apply(int) >=90), "Match/No_Match"] = 1

	#'doc_charge_instructions'
	# post processing fuzzy match percentage
	df.loc[df["Accuracy"].apply(float) >= 90, "Match/No_Match"] = 1
	print(df["Match/No_Match"].sum())
	print(df)
	# exit("++++++++++++")
	res_path= os.path.join(folder_path, 'result_path')
	if not os.path.exists(res_path):
		os.mkdir(res_path)
	df.to_csv(f'{res_path}/CS_analysis_pre_valid_sep_12_after_fuzzy_match_post_processing_latest.csv')

	