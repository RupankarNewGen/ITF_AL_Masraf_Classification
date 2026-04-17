

from typing import List, Dict
import glob
import os
import json
import pandas as pd
from fuzzywuzzy import fuzz
import re

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

def keep_specific_characters(text):
    text = text.split(' ')
    final_text = ''
    for word in text:
        a_text = "".join(ch for ch in word if ch.isalnum() or ch in ',.')
        final_text = final_text + a_text + ' '
    return final_text.strip()

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
    
	pattern = r'\b(?:' + '|'.join(re.escape(term) for term in incoterm_list) + r')\b'

	# Use re.findall to find all matching Incoterms in the text
	found_incoterms = re.findall(pattern, text, re.IGNORECASE)

	# If Incoterms are found, join them with spaces; otherwise, return the complete text
	result = " ".join(found_incoterms) if found_incoterms else text

	return result
  
def generic_date(x):
	"""
    1) should not start and end with any special characters
    2) should have specific length
    :return:
    """
	start = 0
	end = -1
	# remove all special character until you get any number
	# this we need to do from starting as well as beginning
	while x[start].isalnum():
		x = "".join(x[start])
		start += 1

	while x[end].isalnum():
		x = "".join(x[:end])
		end -= 1
	return x

def pp_cash_drawn_rules(x):
    """
    just take 522 or 600 from the string
    :param x:
    :return:
    """
    x = str(x)
    pattern: str = "522|600"
    match1 = re.search(pattern, x)
    return match1[0] if match1 else ""

def remove_symbols(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()  # Strip spaces from the start and end of the text
    return cleaned_text


def generic_page_no(x):
	"""
    remove if any alphabetic characters are there in this
    # 1) if of is present in ground truth , split on of and get the first element and strip that string.
    :param x:
    :return:
    """
	x = str(x)
	x = x.lower()
	if "of" not in x:
		return x.strip()
	list_page_nos: List = x.split("of")

	# take the first element of list_page_nos
	first_element = list_page_nos[0] if list_page_nos else x

	return first_element.strip()


def generic_address(x):
    countries_list = []
    """
    remove the country from the address
    :param x:
    :return:
    """
    address, coord = x
    address = address.lower()
    for country in countries_list:
        if address.__contains__(country):
            address = address.replace(country, "")

    return [address, coord]


def generic_bic(x):
	"""
    1) should have some fixed length string - may be 12 digit or something
    2) does not contain any special character
    :return:
    """
	# Need to be implemented
	pass



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

def extract_currency_and_amount(input_string):
    curr = ''
    amt = ''
    # input_string = re.sub(r'\s+', '', input_string)
    # input_string = re.sub(r'(?<! ) +| +(?= )', '', input_string)
    input_string = re.sub(r'\s+', ' ', input_string)

    characters_to_remove = "'·!'|:()/-%;'*"     #('')
    translation_table = str.maketrans('', '', characters_to_remove)
    input_string = input_string.translate(translation_table) 
    list_of_input_string = input_string.split(' ')
    for j in list_of_input_string:
        flag = None
        for i in j:
            try:
                if int(i):
                    amt+=i
                    flag = 'amt'
                if i=='0':
                    amt+='0'
            except:
                if  i=='.' or i==',':
                    if flag == 'amt':
                        amt+=i
                    elif flag == 'curr':
                        curr+=i 
                    
                else:   
                    curr+=i 
                    flag = 'curr'
        if len(list_of_input_string)>1:
            curr = curr+' '
    return curr, amt


def remove_special_chars(text):
    # Define the pattern for special characters
    pattern = r'^[\W_]+|[\W_]+$'
    # Remove special characters from start and end of text
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text




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


def remove_by_prefix(input_string):
    # Split the input string by space and remove the first element ('by')
    words = input_string.split(' ')
    if words[0].lower() == 'by':
        words = words[1:]
    return ' '.join(words)



def clean_date(original_string):
	# pattern = r'[^a-zA-Z0-9,./-]'
	pattern = r'[^a-zA-Z0-9,./\s-]'
	# Use re.sub() to replace the matched special characters with an empty string
	cleaned_string = re.sub(pattern, '', original_string)
	return cleaned_string


def doc_delivery_pp(input_string):
	# input_string = "RELEASE DOCUMENTS Against PAYMENT"

	# Define a regex pattern to match either "AGAINST PAYMENT" or "AGAINST ACCEPTANCE" (case-insensitive)
	pattern = r'(?i)AGAINST (PAYMENT|ACCEPTANCE)'

	# Use re.search with the IGNORECASE flag to find the pattern in the input string
	match = re.search(pattern, input_string)

	# Check if a match is found
	if match:
		# Extract the matched text
		extracted_text = match.group()
		print(extracted_text)
	else:
		# If no match is found, return the entire input string
		extracted_text = input_string
		print("Pattern not found in the input string. Returning the entire string:")
		print(extracted_text)
	return extracted_text



def doc_chages_pp(input_string):
	target_words = ["buyer's", "drawee", "drawer", "DRAWEES", "DRAWEE'S"]

	# Create a regular expression pattern to match the target words
	pattern = r'\b(?:' + '|'.join(re.escape(word) for word in target_words) + r')\b'

	# Find all matches in the input string
	matches = re.findall(pattern, input_string, re.IGNORECASE)

	# Check if any matches were found
	if matches:
		# If matches were found, print the segregated words
		final_string = 'from '+matches[0]
		print(final_string)
		
	else:
		# If no matches were found, return the entire input string
		print(input_string)
		final_string = input_string
	return final_string


def remove_text_after_phrases(text):
    cleaned_text = re.sub(r'(Ph\.|Tel\.|Zip|Phone)\s*.*', '', text, flags=re.IGNORECASE)	
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def remove_words(text,words_to_remove_list):
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, words_to_remove_list)))
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('&', '')
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def remove_label_name(text,regex):
    # Remove using regular expression    
    match = re.search(regex, text)
    if match:
        text = re.sub(regex, "", text)
        return text.strip()
    return text.strip()






def wrapping_up(key, pre_act_dict, flag):
    sum_insured_currency_lis = []
    # print(pre_act_dict[key])
    for i in range(len(pre_act_dict[key])):
        if type(pre_act_dict[key][i][0]) is list:
            # print('???????????????', labels['sum_insured_currency'][i][0])
            for j in pre_act_dict[key][i][0]:
                # print('jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj', j)
                sum_insured_currency_lis.append(j)
                bbox = pre_act_dict[key][i][1]
                if flag =='pred':
                    confi = pre_act_dict[key][i][2]
        else:
            sum_insured_currency_lis.append(pre_act_dict[key][i][0])
            bbox = pre_act_dict[key][i][1]
            if flag =='pred':
                confi = pre_act_dict[key][i][2]
    if flag == 'actual':
        pre_act_dict[key]=[sum_insured_currency_lis, bbox]
    elif flag == 'pred':
        pre_act_dict[key]=[sum_insured_currency_lis, bbox, confi]
    return pre_act_dict

def currency_amount_segregation(pred_actual_list, data_json, child_key1, child_key2, parent_field = None, prediction_flag = None):
	'''
		pred_actual_list =  list of keys in prediction dictnary or actual dictnary
		parent_field = "currency_amount"
		child_key1 = "csh_bill_amount"
		child_key2 = "csh_bill_currency"
		##### child_key1 or child_key2 may overlap need to check properly before assigning ######
		
	'''
	print(child_key1, child_key2, parent_field)
	if parent_field:
		print(data_json)
		if parent_field in pred_actual_list:
			for i in range(len(data_json[parent_field])):
				act_currency, act_amount = extract_currency_and_amount(str(data_json[parent_field][i][0]))
				if len(act_amount)>0 and child_key1 in pred_actual_list:
					if prediction_flag:
						data_json[child_key1].append([act_amount, data_json[parent_field][i][1], data_json[parent_field][i][2]])

					else:
						data_json[child_key1].append([act_amount, data_json[parent_field][i][1]])
					# data_json[child_key1].append(data_json[parent_field][i][1])
				else:
					if len(act_amount)>0:
						if prediction_flag:
							data_json[child_key1]=[[act_amount, data_json[parent_field][i][1], data_json[parent_field][i][2]]]
						else:
							data_json[child_key1]=[[act_amount, data_json[parent_field][i][1]]]


				if len(act_currency)>0 and child_key2 in pred_actual_list:
					if prediction_flag:
						data_json[child_key2].append([act_currency, data_json[parent_field][i][1], data_json[parent_field][i][2]])

					else:
						data_json[child_key2].append([act_currency, data_json[parent_field][i][1]])
						
					# data_json[child_key1].append(data_json[parent_field][i][1])
				else:
					if len(act_currency)>0:
						if prediction_flag:
							data_json[child_key2]=[[act_currency, data_json[parent_field][i][1], data_json[parent_field][i][2]]]

						else:
							data_json[child_key2]=[[act_currency, data_json[parent_field][i][1]]]
			print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
			print(data_json)
	if child_key2 in pred_actual_list:
		# index_to_delete = []
		for i in range(len(data_json[child_key2])):
			act_currency, act_amount = extract_currency_and_amount(str(data_json[child_key2][i][0]))
			if len(act_amount)>0 and child_key1 in pred_actual_list:
				if prediction_flag:
					data_json[child_key1].append([act_amount, data_json[child_key2][i][1], data_json[child_key2][i][2]])
				else:
					data_json[child_key1].append([act_amount, data_json[child_key2][i][1]])

				# data_json[child_key1].append(data_json[parent_field][i][1])
			else:
				if len(act_amount)>0:
					if prediction_flag:
						data_json[child_key1]=[[act_amount, data_json[child_key2][i][1], data_json[child_key2][i][2]]]

					else:
						data_json[child_key1]=[[act_amount, data_json[child_key2][i][1]]]


			if len(act_currency)>0: #and child_key2 in pred_actual_list:
				data_json[child_key2][i][0] = act_currency   

	print(data_json)

	if child_key1 in pred_actual_list:
		for i in range(len(data_json[child_key1])):
			act_currency, act_amount = extract_currency_and_amount(str(data_json[child_key1][i][0]))
			if len(act_currency)>0 and child_key2 in pred_actual_list:
				if prediction_flag:
					data_json[child_key2].append([act_currency, data_json[child_key1][i][1], data_json[child_key1][i][2]])
				else:
					data_json[child_key2].append([act_currency, data_json[child_key1][i][1]])
				# data_json[child_key1].append(data_json[parent_field][i][1])
			else:
				if len(act_currency)>0:
					if prediction_flag:
						data_json[child_key2]=[[act_currency, data_json[child_key1][i][1], data_json[child_key1][i][2]]]

					else:
						data_json[child_key2]=[[act_currency, data_json[child_key1][i][1]]]


			if len(act_amount)>0: #and child_key2 in pred_actual_list:
				data_json[child_key1][i][0] = act_amount
	print(data_json)
 
	return data_json



def amount_currency_overlapping(key1, key2, labels, predicted):
	actual_list = list(labels)
	pred_list = list(predicted)
	if key1 in actual_list:
		index_to_delete = []
		for i in range(len(labels[key1])):
			amount_list = list()
			sum_insured = labels[key1][i][0].split()
			for j in sum_insured:
				act_currency, act_amount = extract_currency_and_amount(str(j))
				if len(act_currency)>0 and key2 in actual_list:
					labels[key2].append([act_currency, labels[key1][i][1]])
					# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
				else:
					if len(act_currency)>0:
						labels[key2]=[[act_currency, labels[key1][i][1]]]
				if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
					amount_list.append(act_amount)
			if len(amount_list)>0:
				labels[key1][i][0] = amount_list
			else:
				index_to_delete.append(i)
		if len(index_to_delete)>0:
			for d in index_to_delete:
				del labels[key1][d]
			if len(labels[key1])==0:
				del labels[key1]
				# del labels[key1][i]
			#else => needed to delete the i th element in the sum_insured_amount , if required.
		
  
	if key1 in pred_list:
		index_to_delete = []
		for i in range(len(predicted[key1])):
			amount_list = []
			sum_insured = predicted[key1][i][0].split()
			for j in sum_insured:
				act_currency, act_amount = extract_currency_and_amount(str(j))
				if len(act_currency)>0 and key2 in pred_list:
					predicted[key2].append([act_currency, predicted[key1][i][1], predicted[key1][i][2]])
					# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
				else:
					if len(act_currency)>0:
						predicted[key2]=[[act_currency, predicted[key1][i][1], predicted[key1][i][2]]]

				
				if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
					amount_list.append(act_amount)
			if len(amount_list)>0:
				predicted[key1][i][0] = amount_list
			else:
				index_to_delete.append(i)
		if len(index_to_delete)>0:
			for d in index_to_delete:
				del predicted[key1][d]
			if len(predicted[key1])==0:
				del predicted[key1]


	if key2 in actual_list:
		index_to_delete = []
		for i in range(len(labels[key2])):
			currency_list = []
			sum_insured = labels[key2][i][0].split()
			for j in sum_insured:
				act_currency, act_amount = extract_currency_and_amount(str(j))
				if len(act_amount)>0 and key1 in actual_list:
					labels[key1].append([act_amount, labels[key2][i][1]])
					# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
				else:
					if len(act_amount)>0:
						labels[key1]=[[act_amount, labels[key2][i][1]]]

				
				if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
					currency_list.append(act_currency)
					
																																																																										
			if len(currency_list)>0:
				labels[key2][i][0] = currency_list

			else:
				index_to_delete.append(i)
		if len(index_to_delete)>0:
			for d in index_to_delete:
				del labels[key2][d]
			if len(labels[key2])==0:
				del labels[key2]
				
	if key2 in pred_list:
		index_to_delete = []
		for i in range(len(predicted[key2])):
			currency_list = []
			sum_insured = predicted[key2][i][0].split()
			for j in sum_insured:
				act_currency, act_amount = extract_currency_and_amount(str(j))
				if len(act_amount)>0 and key1 in pred_list:
					predicted[key1].append([act_amount, predicted[key2][i][1], predicted[key2][i][2]])
					# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
				else:
					if len(act_amount)>0:
						predicted[key1]=[[act_amount, predicted[key2][i][1], predicted[key2][i][2]]]

				
				if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
					currency_list.append(act_currency)
				
			if len(currency_list)>0:
				predicted[key2][i][0] = currency_list
			else:
				index_to_delete.append(i)
		if len(index_to_delete)>0:
			for d in index_to_delete:
				del predicted[key2][d]
			if len(predicted[key2])==0:
				del predicted[key2]
	if key1 in actual_list:
		labels = wrapping_up(key1, labels, 'actual')
	if key2 in actual_list:
		labels = wrapping_up(key2, labels, 'actual')
  
	if key1 in pred_list:
		predicted = wrapping_up(key1, predicted, 'pred')
  
	if key2 in pred_list:
		predicted = wrapping_up(key2, predicted, 'pred')
  
	return labels, predicted


def handle_cs_amount_currency(labels, predicted):
	actual_list = list(labels)
	pred_list = list(predicted)
	print(actual_list,'*****',pred_list)		
	if 'currency_amount' in actual_list:
		for i in range(len(labels['currency_amount'])):
			act_currency, act_amount = extract_currency_and_amount(str(labels['currency_amount'][i][0]))
			if len(act_amount)>0 and 'csh_bill_amount' in actual_list:
				labels['csh_bill_amount'].append([act_amount, labels['currency_amount'][i][1]])
				# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
			else:
				if len(act_amount)>0:
					labels['csh_bill_amount']=[[act_amount, labels['currency_amount'][i][1]]]


			if len(act_currency)>0 and 'csh_bill_currency' in actual_list:
				labels['csh_bill_currency'].append([act_currency, labels['currency_amount'][i][1]])
				# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
			else:
				if len(act_currency)>0:
					labels['csh_bill_currency']=[[act_currency, labels['currency_amount'][i][1]]]

	if 'currency_amount' in pred_list:
		for i in range(len(predicted['currency_amount'])):
			act_currency, act_amount = extract_currency_and_amount(str(predicted['currency_amount'][i][0]))
			if len(act_amount)>0 and 'csh_bill_amount' in pred_list:
				predicted['csh_bill_amount'].append([act_amount, predicted['currency_amount'][i][1], predicted['currency_amount'][i][2]])
				# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
			else:
				if len(act_amount)>0:
					predicted['csh_bill_amount']=[[act_amount, predicted['currency_amount'][i][1], predicted['currency_amount'][i][2]]]


			if len(act_currency)>0 and 'csh_bill_currency' in pred_list:
				predicted['csh_bill_currency'].append([act_currency, predicted['currency_amount'][i][1], predicted['currency_amount'][i][2]])
				# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
			else:
				if len(act_currency)>0:
					predicted['csh_bill_currency']=[[act_currency, predicted['currency_amount'][i][1], predicted['currency_amount'][i][2]]]

	if 'csh_bill_currency' in actual_list:
		for i in range(len(labels['csh_bill_currency'])):
			act_currency, act_amount = extract_currency_and_amount(str(labels['csh_bill_currency'][i][0]))
			if len(act_amount)>0 and 'csh_bill_amount' in actual_list:
				labels['csh_bill_amount'].append([act_amount, labels['csh_bill_currency'][i][1]])
				# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
			else:
				if len(act_amount)>0:
					labels['csh_bill_amount']=[[act_amount, labels['csh_bill_currency'][i][1]]]


			if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
				labels['csh_bill_currency'][i][0] = act_currency   


	if 'csh_bill_currency' in pred_list:
		for i in range(len(predicted['csh_bill_currency'])):
			act_currency, act_amount = extract_currency_and_amount(str(predicted['csh_bill_currency'][i][0]))
			if len(act_amount)>0 and 'csh_bill_amount' in pred_list:
				predicted['csh_bill_amount'].append([act_amount, predicted['csh_bill_currency'][i][1], predicted['csh_bill_currency'][i][2]])
				# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
			else:
				if len(act_amount)>0:
					predicted['csh_bill_amount']=[[act_amount, predicted['csh_bill_currency'][i][1], predicted['csh_bill_currency'][i][2]]]

			if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
				predicted['csh_bill_currency'][i][0] = act_currency   


	if 'csh_bill_amount' in actual_list:
		for i in range(len(labels['csh_bill_amount'])):
			act_currency, act_amount = extract_currency_and_amount(str(labels['csh_bill_amount'][i][0]))
			if len(act_currency)>0 and 'csh_bill_currency' in actual_list:
				labels['csh_bill_currency'].append([act_currency, labels['csh_bill_amount'][i][1]])
				# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
			else:
				if len(act_currency)>0:
					labels['csh_bill_currency']=[[act_currency, labels['csh_bill_amount'][i][1]]]


			if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
				labels['csh_bill_amount'][i][0] = act_amount   


	if 'csh_bill_amount' in pred_list:
		for i in range(len(predicted['csh_bill_amount'])):
			act_currency, act_amount = extract_currency_and_amount(str(predicted['csh_bill_amount'][i][0]))
			if len(act_currency)>0 and 'csh_bill_currency' in pred_list:
				predicted['csh_bill_currency'].append([act_currency, predicted['csh_bill_amount'][i][1], predicted['csh_bill_amount'][i][2]])
				# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
			else:
				if len(act_currency)>0:
					predicted['csh_bill_currency']=[[act_currency, predicted['csh_bill_amount'][i][1], predicted['csh_bill_amount'][i][2]]]


			if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
				predicted['csh_bill_amount'][i][0] = act_amount     
		print(predicted)
		print("*****************")

	return labels, predicted





def handle_ic_currency_amt(labels, predicted):
	actual_list = list(labels)
	pred_list = list(predicted)
	if 'sum_insured_amount' in actual_list:
			# exit('???????????????????????????????????')
		index_to_delete = []
		for i in range(len(labels['sum_insured_amount'])):
			amount_list = list()
			sum_insured = labels['sum_insured_amount'][i][0].split()
			for j in sum_insured:
				act_currency, act_amount = extract_currency_and_amount(str(j))
				if len(act_currency)>0 and 'sum_insured_currency' in actual_list:
					labels['sum_insured_currency'].append([act_currency, labels['sum_insured_amount'][i][1]])
					# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
				else:
					if len(act_currency)>0:
						labels['sum_insured_currency']=[[act_currency, labels['sum_insured_amount'][i][1]]]
				if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
					amount_list.append(act_amount)
			if len(amount_list)>0:
				labels['sum_insured_amount'][i][0] = amount_list
			else:
				index_to_delete.append(i)
		if len(index_to_delete)>0:
			for d in index_to_delete:
				del labels['sum_insured_amount'][d]
			if len(labels['sum_insured_amount'])==0:
				del labels['sum_insured_amount']
				# del labels['sum_insured_amount'][i]
			#else => needed to delete the i th element in the sum_insured_amount , if required.
	# if 'sum_insured_amount' in actual_list:

	if 'sum_insured_amount' in pred_list:
		index_to_delete = []
		for i in range(len(predicted['sum_insured_amount'])):
			amount_list = []
			sum_insured = predicted['sum_insured_amount'][i][0].split()
			for j in sum_insured:
				act_currency, act_amount = extract_currency_and_amount(str(j))
				if len(act_currency)>0 and 'sum_insured_currency' in pred_list:
					predicted['sum_insured_currency'].append([act_currency, predicted['sum_insured_amount'][i][1], predicted['sum_insured_amount'][i][2]])
					# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
				else:
					if len(act_currency)>0:
						predicted['sum_insured_currency']=[[act_currency, predicted['sum_insured_amount'][i][1], predicted['sum_insured_amount'][i][2]]]
	
				
				if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
					amount_list.append(act_amount)
			if len(amount_list)>0:
				predicted['sum_insured_amount'][i][0] = amount_list
			else:
				index_to_delete.append(i)
		if len(index_to_delete)>0:
			for d in index_to_delete:
				del predicted['sum_insured_amount'][d]
			if len(predicted['sum_insured_amount'])==0:
				del predicted['sum_insured_amount']


	if 'sum_insured_currency' in actual_list:
		index_to_delete = []
		for i in range(len(labels['sum_insured_currency'])):
			# if file[0:-11]=='Insurance_Certificate_34_page_0':
			# 	print(labels['sum_insured_currency'])
				# exit()
			currency_list = []
			sum_insured = labels['sum_insured_currency'][i][0].split()
			# if file[0:-11]=='Insurance_Certificate_34_page_0':
			# 	print(labels['sum_insured_currency'])
			# 	print(sum_insured)
				# exit()
			for j in sum_insured:
				act_currency, act_amount = extract_currency_and_amount(str(j))
				if len(act_amount)>0 and 'sum_insured_amount' in actual_list:
					labels['sum_insured_amount'].append([act_amount, labels['sum_insured_currency'][i][1]])
					# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
				else:
					if len(act_amount)>0:
						labels['sum_insured_amount']=[[act_amount, labels['sum_insured_currency'][i][1]]]
	
				
				if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
					currency_list.append(act_currency)
					
																																																																										
			if len(currency_list)>0:
				labels['sum_insured_currency'][i][0] = currency_list

			else:
				index_to_delete.append(i)
		if len(index_to_delete)>0:
			for d in index_to_delete:
				del labels['sum_insured_currency'][d]
			if len(labels['sum_insured_currency'])==0:
				del labels['sum_insured_currency']
				
	if 'sum_insured_currency' in pred_list:
		index_to_delete = []
		for i in range(len(predicted['sum_insured_currency'])):
			currency_list = []
			sum_insured = predicted['sum_insured_currency'][i][0].split()
			for j in sum_insured:
				act_currency, act_amount = extract_currency_and_amount(str(j))
				if len(act_amount)>0 and 'sum_insured_amount' in pred_list:
					predicted['sum_insured_amount'].append([act_amount, predicted['sum_insured_currency'][i][1], predicted['sum_insured_currency'][i][2]])
					# labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
				else:
					if len(act_amount)>0:
						predicted['sum_insured_amount']=[[act_amount, predicted['sum_insured_currency'][i][1], predicted['sum_insured_currency'][i][2]]]
	
				
				if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
					currency_list.append(act_currency)
				
			if len(currency_list)>0:
				predicted['sum_insured_currency'][i][0] = currency_list
			else:
				index_to_delete.append(i)
		if len(index_to_delete)>0:
			for d in index_to_delete:
				del predicted['sum_insured_currency'][d]
			if len(predicted['sum_insured_currency'])==0:
				del predicted['sum_insured_currency']
			#else => needed to delete the i th element in the sum_insured_currency if there is nothing found in currency , but it is present in the prediction 
			# eg: 'sum_insured_currency' : 2,340 => it will be filterd as sum_insured_amount and sum_insured_currency will be empty , 
			# in this case we need to delete i th element in sum_insured_currency
			#else => needed to delete the i th element in the sum_insured_amount , if required.(wrong)
			#sample:
			'''Insurance_Certificate_1_page_0_labels.txt
			actual_currency #######################################
			[[['USD'], [392, 1287, 763, 1342]]]
			predicted_curreny #######################################
			[['( 3 )', [390, 1287, 416, 1308], 66.34], [['usd'], [718, 1319, 759, 1336], 96.96]]
			'( 3 )'==> got filtered and appended to the amount  and remain nothing in currency, as we are not deleting the i th element it remain as it is
			predicted_amount ####################################
			[[['314600', '4840'], [660, 1291, 736, 1336], 49.89], ['3', [390, 1287, 416, 1308]]]'''
	if 'sum_insured_amount' in list(labels):
		print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx', labels['sum_insured_amount'])
		
		labels = wrapping_up('sum_insured_amount', labels, 'actual')
		print('actual #######################################')
		print('actual #######################################')
		print('actual #######################################')
		print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy', labels['sum_insured_amount'])
		
		
	if 'sum_insured_currency' in list(labels):
		labels = wrapping_up('sum_insured_currency', labels, 'actual')
		print('actual #######################################')
		print('actual #######################################')
		print('actual #######################################')
		print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy',labels['sum_insured_currency'])   
		# print(labels['sum_insured_amount'])
		
		#[[['060.00'], [1310, 692, 1432, 714]], ['49', [1269, 690, 1309, 714]]]
	if 'sum_insured_amount' in list(predicted):
		predicted = wrapping_up('sum_insured_amount', predicted, 'pred')
		print('actual #######################################')
		print('actual #######################################')
		print('actual #######################################')
		print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy', predicted['sum_insured_amount'])
		
		
	if 'sum_insured_currency' in list(predicted):
		predicted = wrapping_up('sum_insured_currency', predicted, 'pred')
		print('actual #######################################')
		print('actual #######################################')
		print('actual #######################################')
		print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy',predicted['sum_insured_currency']) 

	return labels, predicted
		
		