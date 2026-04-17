from typing import List
import os
import json
import csv
from bisect import bisect_left
import re

"""
Sample master data json:

{"drawer_bank_name": [["HSBC BANK MALAYSIA BERHAD", [260, 311, 686, 337]]], 
"drawer_bank_address": [["2 LEBOH AMPANG 50100 KUALA LUMPUR", [256, 346, 569, 407]], 
["KUALA LUMPUR MAIN", [260, 278, 557, 304]], 
["MALAYSIA", [257, 415, 405, 442]]], "page_no": [["1", [1255, 213, 1287, 239]]], "drawee_bank_name": [["ICICI BANK LTD", [257, 548, 508, 577]]], "drawee_bank_address": [["REGIONAL TRADE SVCS UNIT - NEW DEHI 9A PHELPS BUILDING , CONNAUGHT PLACE", [259, 582, 860, 646]], ["NEW DELHI 110001", [261, 654, 531, 681]], ["INDIA", [542, 652, 638, 682]]], "csh_presentation_date": [["19FEB2016", [1197, 685, 1356, 715]]], "drawer_name": [["INTERPRINT DECOR MALAYSIA SDN BHD", [444, 754, 1003, 780]]], "drawee_name": [["GREENLAM INDUSTRIES LIMITED", [444, 787, 904, 814]]], "drawee_address": [["DISTT SOLAN , HIMACHAL PRADES ,", [442, 821, 930, 854]], ["INDIA", [942, 820, 1038, 853]]], "csh_bill_currency": [["EUR", [630, 1093, 683, 1123]]], "csh_bill_amount": [["42,026.57", [840, 1093, 1002, 1126]]], "usance_tenor": [["180", [443, 924, 496, 950]]], "tenor_indicator": [["DAYS", [527, 925, 605, 950]]], "tenor_indicator_type": [["FROM", [615, 922, 685, 951]]], "tenor_indicator_date": [["INVOICE DATE", [697, 920, 903, 951]]], "csh_ref_no": [["OBCKLH770358", [958, 614, 1169, 648]]], "csh_drawn_under_rules": [["522", [1023, 1532, 1086, 1565]]], "doc_charge_instructions": [["COLLECT YOUR CHARGES AND EXPENSES FROM DRAWEE", [278, 1767, 1036, 1797]]], "doc_delivery_instruction": [["* ACCEPTANCE / PAYMENT MAY BE DEFERRED PENDING ARRIVAL OF CARRYING VESSEL .", [261, 1702, 1173, 1759]]], 
"doc_settlement_instructions": [["HSBC BANK PLC ,", [1074, 2039, 1312, 2068]]]}
"""

folder_path: str = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/May-8-Certificate-Of-Origin/data/complete_data"
# try: 
# 	with open(os.path.join(folder_path, 'codes-all.csv'), 'r') as file:
# 		my_reader = csv.reader(file, delimiter=',')
# 		currency_list = [row[2] for row in my_reader if len(row) > 2]

# except Exception as e:
# 	print(e)
# 	print(f"The file unable to read pls check the file path")

# # correcting the currency list
# currency_list.remove("AlphabeticCode")
# currency_list.remove("")

# print(currency_list)


def binary_search(a, x, lo=0, hi=None):
	if hi is None:
		hi = len(a)
	pos = bisect_left(a, x, lo, hi)
	return pos if pos != hi and a[pos].__contains__(x) else -1


def pp_amount(x):
	"""
    Input should be the merged currency field. The idea is if we are getting the scattered token for currency amount,
    in prediction, need to merge those and then post-processing is applied on it.
    you need to take the currency from the combined currency and
    Solution: 1) may be you can remove the stop words such as comma, we cannot remove the dot because it can change the
    value of the amount.
    2) other that we need to remove the currency portion from this and take the amount separately from this.

    :param x:
    :return:
    """
	string_after_removing_stop_words: str = ''.join(e for e in x if e.isalnum())
	# now take out currency from the string
	# assuming we should have a lookup for the currency
	# i think we can have currency list in one text file

	# binary search for matching string
	# sort the currency list
	index_of_matching_currency = binary_search(currency_list, string_after_removing_stop_words)

	string_after_currency_removal: str = string_after_removing_stop_words.replace(
		currency_list[index_of_matching_currency], ""
	)

	return string_after_currency_removal


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


if __name__ == "__main__":
	variable_to_post_process_with_corresponding_functions: dict = {"currency_amount": pp_amount}
	folder_path: str = ""
	files: List = os.listdir(folder_path)
	for file in files:
		complete_file_path: str = os.path.join(folder_path, file)

		# reading the text file as json
		with open(os.path.join(folder_path, f"{file}_text.txt"), "r") as f:
			master_data_json = json.load(f)

		for key, func_name in variable_to_post_process_with_corresponding_functions.items():
			# fetch the data from the master data json for the corresponding key
			value_corresponding_for_each_key: List[List] = master_data_json[key]

			# now you need to iterate over all values in this list
			# apply post-processing
			value_corresponding_for_each_key = list(map(func_name,
			                                            value_corresponding_for_each_key))
			master_data_json[key] = value_corresponding_for_each_key
