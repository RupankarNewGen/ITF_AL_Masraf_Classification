

import os

import glob
import json
# from accuracy_gen_script_3 import *
import glob
from typing import List
import os
import json
from configparser import ConfigParser
import json
from configparser import ConfigParser
config_ = ConfigParser()
config_.read("post_processing/config.ini")
from label_category import LabelCategory
from fuzzywuzzy import fuzz
from utility import *
from pre_mapping import PredictionKeyMapping, ParentKeyMapping, OverLappingKeys, \
    transport_fields, weights_fields, amount_fields, currency_fields, \
        bottom_value, top_value

match_name = "Match/No_Match"

def load_data(result_path, data_path):
    occurrences = {}
    data = []

    # folder_path: str = "/home/ntlpt-42/Documents/mani_projects/IDP/Trade_finance_end_to_end_test/Final_data_training_layoutlmv2/PL/val_data"
    # folder_path: str = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/Layoutlmv3_performance_phase_1_data/CS_jul_13/evaluation_data"

    # result_path = os.path.join(folder_path, "Results_BOL_validated")
    # data_path = os.path.join(folder_path, "New_Master_Data_Merged")

    # import json
    data_files = os.listdir(data_path)
    result_files = os.listdir(result_path)

    # with open("incoterm_list.txt") as fp:
    # 	for line in fp:
    # 		incoterm_list.append(line.strip())
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

    return data_files, result_files



"""
Sample master internal_data json:

{"drawer_bank_name": [["HSBC BANK MALAYSIA BERHAD", [260, 311, 686, 337]]], 
"drawer_bank_address": [["2 LEBOH AMPANG 50100 KUALA LUMPUR", [256, 346, 569, 407]], 
["KUALA LUMPUR MAIN", [260, 278, 557, 304]], 
["MALAYSIA", [257, 415, 405, 442]]], "page_no": [["1", [1255, 213, 1287, 239]]], "drawee_bank_name": [["ICICI BANK LTD", [257, 548, 508, 577]]], "drawee_bank_address": [["REGIONAL TRADE SVCS UNIT - NEW DEHI 9A PHELPS BUILDING , CONNAUGHT PLACE", [259, 582, 860, 646]], ["NEW DELHI 110001", [261, 654, 531, 681]], ["INDIA", [542, 652, 638, 682]]], "csh_presentation_date": [["19FEB2016", [1197, 685, 1356, 715]]], "drawer_name": [["INTERPRINT DECOR MALAYSIA SDN BHD", [444, 754, 1003, 780]]], "drawee_name": [["GREENLAM INDUSTRIES LIMITED", [444, 787, 904, 814]]], "drawee_address": [["DISTT SOLAN , HIMACHAL PRADES ,", [442, 821, 930, 854]], ["INDIA", [942, 820, 1038, 853]]], "csh_bill_currency": [["EUR", [630, 1093, 683, 1123]]], "csh_bill_amount": [["42,026.57", [840, 1093, 1002, 1126]]], "usance_tenor": [["180", [443, 924, 496, 950]]], "tenor_indicator": [["DAYS", [527, 925, 605, 950]]], "tenor_indicator_type": [["FROM", [615, 922, 685, 951]]], "tenor_indicator_date": [["INVOICE DATE", [697, 920, 903, 951]]], "csh_ref_no": [["OBCKLH770358", [958, 614, 1169, 648]]], "csh_drawn_under_rules": [["522", [1023, 1532, 1086, 1565]]], "doc_charge_instructions": [["COLLECT YOUR CHARGES AND EXPENSES FROM DRAWEE", [278, 1767, 1036, 1797]]], "doc_delivery_instruction": [["* ACCEPTANCE / PAYMENT MAY BE DEFERRED PENDING ARRIVAL OF CARRYING VESSEL .", [261, 1702, 1173, 1759]]], 
"doc_settlement_instructions": [["HSBC BANK PLC ,", [1074, 2039, 1312, 2068]]]}
"""

######## need to extract from the txt file !!!!

class LookUp:
    def __init__(self, incoterm_path, countries_path) -> None:
        self.incoterm_path = incoterm_path
        self.countries_path = countries_path
        self.incoterm_list: list = []
        with open(self.incoterm_path, 'r') as file:
            for line in file:
                 self.incoterm_list.append(line.strip())
        
        self.countries_list: list = []
        with open(self.countries_path, 'r') as file:
            for line in file:
                 self.countries_list.append(line.strip())
        

    def process_incoterm():
        pass




def check_and_modify_labels(json_data, bottom_value,label):
    if label in json_data and bottom_value in json_data:
        json_data[bottom_value][0][0] = "~ "+json_data[bottom_value][0][0]
        
        #joined_value = "~ ".join([json_data[label][0][0], json_data["insurance_issuer_address_bottom"][0][0]])
        joined_value = json_data[label]+json_data[bottom_value]
        json_data[label] = joined_value
        del json_data[bottom_value]
    else:
        if bottom_value in json_data:
            json_data[label] = json_data.pop(bottom_value)

    return json_data


def pre_handling_overlapping(labels, predicted, doc_class):
    print(doc_class)
    if doc_class in ParentKeyMapping:
        parent_child_dict = ParentKeyMapping[doc_class]
        for parent_key, child_key in parent_child_dict.items():
            pass
            # labels = currency_amount_segregation(list(labels), labels, child_key[0], child_key[1], parent_field = parent_key, prediction_flag = None)
            # predicted = currency_amount_segregation(list(predicted), predicted, child_key[0], child_key[1], parent_field = parent_key, prediction_flag = True)
    if doc_class == "CS":
        labels, predicted = handle_cs_amount_currency(labels, predicted)
    ###################################################
    # if doc_class == "IC":
    #     labels, predicted = handle_ic_currency_amt(labels, predicted)
    ###################################################
    
    # if doc_class in OverLappingKeys:
    #     overlapping_dict = OverLappingKeys[doc_class]
    #     for key1, key2 in overlapping_dict.items():
    #         labels, predicted = amount_currency_overlapping(key1, key2, labels, predicted)
    return labels, predicted


def handle_top_bottom_values(labels, predicted, doc_class):
    print(len(top_value))
    print(len(bottom_value))
    if doc_class in PredictionKeyMapping:
        keys_dict = PredictionKeyMapping[doc_class]
        for key, value in keys_dict.items():
            if value not in labels and value in predicted and key in labels:
                predicted = check_and_modify_labels(predicted, value, key)
    
    
    if len(top_value)== len(bottom_value):
        for i in range(len(top_value)):
            labels = check_and_modify_labels(labels,bottom_value[i], top_value[i])
            predicted = check_and_modify_labels(predicted,bottom_value[i], top_value[i])
    else:
        print('The len of top and bottom values should be same')
    return labels, predicted


def filter_prediction(new_row, actual_value, predicted, to_do, data, ratio_):
    l2 = len(predicted)
    print(predicted)
    print(f'The no of unique predictions: {l2}')
    print(f'The no of actual_value >>>>>>>>>>>>>>> : {actual_value}')
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
        # actual= actual.replace(' ','')
        pred= predicted[j][0].lower()
        # predicted= predicted.strip()
        # pred= pred.replace(' ','')

        print(f'The actual value inside for loop : {actual_value[0].lower(), type(actual_value[0].lower())}')
        print(f'the prediction value inside for loop: {predicted[0][0].lower(), type(predicted[0][0].lower())}')
        # if (actual.lower())== (pred.lower()) and flag==True:
        if fuzz.ratio(str(actual).lower(), str(pred).lower()) >= ratio_:
            print('entered into filter +++++++++++++++++===')
            flag= False                                   # consider one prediction 
            new_row.append(actual_value[0])
            new_row.append(predicted[j][0])
            accuracy = fuzz.ratio(str(actual).lower(), str(pred).lower())
            new_row.append(accuracy)
            if accuracy >= ratio_:
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


def final_overall_analysis(path1, path2, folder_path, doc_code, category_name):
	# varaiable definition
	report_path = path1
	analysis_report_path = path2
	accuracy_lookup = pd.read_csv(report_path)
	analysis_report = pd.read_csv(analysis_report_path)
	data = []
	label_names = []

	# final_Report_Summary_Best_Fields
	for index, row in accuracy_lookup.iterrows():
		if row["Complete_Match_Percentage"] >= 80 and row["Fuzzy_Match_Percentage"] >= 85 and row[
			"Label_Name"] != "OVERALL":  # change these metrics to create buckets.
			label_names.append(str(row["Label_Name"]))
			data.append(list(row)[1:])

	if not os.path.exists(os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")):
		os.makedirs(os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}"))


	save_path_1 = os.path.join(folder_path, 'result_path',category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", 
							f"{doc_code}_Final_Report_Summary_Best_Fields_after_fuzzy_match_change_{str(datetime.now())}.csv")

	df2 = pd.DataFrame(data, columns=list(accuracy_lookup.columns)[1:])
	df2.to_csv(save_path_1)

	new_data = [
		list(row)[1:]
		for index, row in analysis_report.iterrows()
		if str(row["label_name"]) in label_names
	]
	save_path_2 = os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", \
							f"{doc_code}_Overall_Analysis_Best_Fields_after_fuzzy_match_change_{str(datetime.now())}.csv")

	df2 = pd.DataFrame(new_data, columns=list(analysis_report.columns)[1:])
	df2.to_csv(save_path_2)


def final_report(csv01, folder_path, doc_code, category_name):
	df = pd.read_csv(os.path.join(csv01))

	report = {}
	label_names = []
	label_counts = []
	label_detected = []
	detection_accuracy = []
	average_accuracy = []
	matched_labels = []
	matched_detected = []
	total_match_percentage = []
	g = df.groupby("label_name")
	for name, name_df in g:
		label_names.append(name)
		label_counts.append(len(name_df.index))
		average_accuracy.append(round(name_df["Accuracy"].mean(), 2))
		matched = sum(list(name_df[match_name]))
		print(f'matched numbers: {matched}')
		matched_labels.append(matched)
		total_match_percentage.append(round((matched / len(name_df.index)) * 100, 2))
		not_detected = name_df["predicted"].isnull().sum()
		detected = len(name_df.index) - not_detected
		print(f'detected numbers: {detected}')
		label_detected.append(detected)
		matched_detected.append(matched / detected)
		print(matched_detected)
		detection_accuracy.append((detected / len(name_df.index)) * 100)
	data = {'Label_Name': label_names, 'Label_Count': label_counts, "Labels_Detected": label_detected,
			"Detection_Accuracy": detection_accuracy, "Fuzzy_Match_Percentage": average_accuracy,
			"Complete_Match_Count_Detected": matched_detected, "Complete_Match_Count": matched_labels,
			"Complete_Match_Percentage": total_match_percentage}

	# Just to calculate detection accuracy
	detected = sum(label_detected)
	all_matched = sum(matched_labels)
	avg_matched_detected = all_matched / detected
	all_labels = sum(label_counts)
	overall_detection_accuracy = (detected / all_labels) * 100
	# dataframe and csv file generation
	report = pd.DataFrame(data)
	li = ["OVERALL", sum(list(report["Label_Count"])), sum(list(report["Labels_Detected"])), overall_detection_accuracy,
		df["Accuracy"].mean(), avg_matched_detected, sum(list(report["Complete_Match_Count"])),
		sum(list(report["Complete_Match_Count"])) / sum(list(report["Label_Count"]))]
	report.loc[len(report.index)] = li

	if not os.path.exists(os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")):
		os.makedirs(os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}"))


	name = f"final_report_{doc_code}_pre_{str(datetime.now())}_after_fuzzy_match_change.csv"

	file_path_csv2 = os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", name)
	report.to_csv(file_path_csv2)

	# txt file generation
	name = f"final_report_{datetime.now()}" + ".txt"
	name_path = os.path.join(folder_path, 'result_path',category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}",  name)
	text_report = {
		row["Label_Name"]: {
			"Label_Count": row["Label_Count"],
			"Detection_Accuracy": row["Detection_Accuracy"],
			"Fuzzy_Match_Percentage": row["Fuzzy_Match_Percentage"],
			"Complete_Match_Count_Detected": row["Complete_Match_Count_Detected"],
			"Complete_Match_Count": row["Complete_Match_Count"],
			"Complete_Match_Percentage": row["Complete_Match_Percentage"],
		}
		for index, row in report.iterrows()
	}
	with open(name_path, "w") as f:
		json.dump(text_report, f)
	f.close()

	return file_path_csv2


def stp_report(csv02, folder_path, doc_code, category_name):
	df = pd.read_csv(csv02)
	g = df.groupby("File_Name")
	stp_data = []
	num_files = 0
	match_files = 0
	for name, name_df in g:
		num_files += 1
		match_value = sum(list(name_df[match_name]))
		num_labels = len(list(name_df[match_name]))
		if match_value == num_labels:
			match_files += 1
			flag = 1
		else:
			flag = 0
		row = [str(name), match_value, num_labels, flag]
		stp_data.append(row)
	print("stp is", (match_files / num_files) * 100)

	df2 = pd.DataFrame(stp_data, columns=["File_Name", "Labels_Matched", "Labels_Present", "STP_Match"])
	file_path_csv3 = os.path.join(folder_path, 'result_path', category_name, 
                               f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", 
                               f"STP_Summary_{datetime.now()}.csv")
	df2.to_csv(file_path_csv3)
 
 
def fuzzy_float_comparison(float1, float2, tolerance=1e-6):
	absolute_difference = abs(float1 - float2)
	print(f'absolute difference: {absolute_difference}')
	if absolute_difference <= tolerance:
		return 100  # Return 100 for a perfect match within the tolerance
	else:
		# Calculate a similarity score based on the relative difference
		similarity = 100 - (absolute_difference / max(abs(float1), abs(float2))) * 100
		return similarity

'''def handle_address_fields(row, fuzzy_ratio=90):
    
    row[2] = filter_address(row[2])
    row[3] = filter_address(row[3])
    print(f'The actual value: {row[2]}')
    print(f'The pred value: {row[3]}')
    # exit('+++++++++++===')
    print('Entered+++++++++++++++++++++=')
    if row[2] and row[3] is not None:
        row[2]= remove_spaces(row[2])
        row[3]=  remove_spaces(row[3])
        
    row = get_fuzz_accuracy(row, fuzzy_ratio)
    
    return row'''


def handle_address_fields(row, index1=None):
    if index1:
        row[index1] = remove_symbols(row[index1])
        row[index1] = remove_spl_char_multiple_pred(row[index1])
        row[index1] = remove_text_after_phrases(filter_address(row[index1]))
        # exit('+++++++++++===')
        print('Entered+++++++++++++++++++++=')
        if row[index1] is not None:
            row[index1]= remove_spaces(row[index1])
    return row

def get_fuzz_accuracy(row, fuzzy_ratio):
    fuzzy_accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
    row.append(fuzzy_accuracy)
    if fuzzy_accuracy >= fuzzy_ratio:
        row.append(1)
    else:
        row.append(0)
        
    return row
        

'''def pre_handle_in_ic(key, labels, predicted, row, data):
    if key in labels and key in predicted:
        actual = labels[key]
        predicted_label = predicted[key]
        if key in ['sum_insured_currency', 'sum_insured_amount']:
            act_val = ''
            pre_val = ''
            act_val = str(actual[0])
            pre_val = str(predicted_label[0])
            print(act_val)
            # exit()
            print(actual)
            # for i in range(len(actual)):
            #     act_val += str(actual[i][0])+", "
            # for j in range(len(predicted_label)):
            #     pre_val += str(predicted_label[j][0])+", "
            row.append(key)    
            new_row1 = row.copy()
            new_row1.append(act_val)
            new_row1.append(pre_val)
                    
            accuracy = fuzz.ratio(str(new_row1[2]).lower(), str(new_row1[3]).lower())
            new_row1.append(accuracy)
            if accuracy == 100:
                new_row1.append(1)
            else:
                new_row1.append(0)
            new_row1.append("predicted[j][2]")
            new_row1.append("actual_value[1]")
            data.append(new_row1)   
    return data'''


def handle_numeric_fields(row, key, index1=None): 
    if index1:
        row[index1] = remove_spl_char_multiple_pred(row[index1])  
        row[index1] = remove_spaces(row[index1])
        
        if key in weights_fields:
            row[index1] = keep_specific_characters(row[index1]) 
        else:
            row[index1] = remove_alphabets(row[index1])
        
    return row

def handle_date_fields(row, index1=None):    
    if index1:
        row[index1] = clean_date(row[index1])

    return row

def handle_single_word_fields(key, row, incoterm_list, index1=None):
    if index1:
        # row[index1] = remove_spl_char_multiple_pred(row[index1])
        if key == "incoterm":
            row[index1] = preprocess_incoterm(row[index1], incoterm_list)
        # if key == "pageno":
        #     row[index1] = generic_page_no(row[index1])
        if key in transport_fields:
            row[index1] = remove_words(row[index1],['BY',"EXPORT", 'freight'])
            row[index1] = remove_by_prefix(row[index1])
        if key == 'csh_drawn_under_rules':
            row[index1] = pp_cash_drawn_rules(row[index1])
        if key =='port_of_discharge':
            regex="([PDo])?(o)?(r)?(t)?(\s)?(o)?(f)?(\s)?([DO0])?(i)?(s)?(c)?(h)?(a)?(r)?(g)?(e)?"
            row[index1] = remove_label_name(row[index1],regex)
        if key =='port_of_loading':
            regex=r"([PDo])?(o)?(r)?(t)?(\s)?(o)?(f)?(\s)?([IL1T\|])?(o)?(a)?(d)?(i)?(n)?(g)?"
            row[index1] = remove_label_name(row[index1],regex)
        if key =='page_no' or key=="original_Number":
            regex = r"(?i)(page|no\.|no|pago\s*no\.|pago|Page)"
            row[index1] = generic_page_no(remove_special_chars(remove_label_name(row[index1],regex)))
        if key =='to_place':
            regex = r"(?i)^To\s*:"
            row[index1] = remove_label_name(row[index1],regex)
        if key =="from_place":										
            row[index1] = remove_words(row[index1],['FROM'])
            
    return row


def handle_single_line_fields(key, row, index1=None):
    if index1:
        row[index1] = remove_spl_char_multiple_pred(row[index1])
        if key == "doc_delivery_instruction":
            row[index1] = doc_delivery_pp(row[index1])
        if key == "doc_charge_instructions":
            row[index1] = doc_chages_pp(row[index1]) 
        if key =='consignee_name':
            regex = r"(?i)NOTIFY\s*:\s*"
            row[index1] = remove_label_name(row[index1],regex)
        if key =="consignor_name":										
            row[index1] = remove_words(row[index1],['EXPORTER'])
        if key =="declaration_by" or key=="declaration":
            row[index1] = remove_words(row[index1],["REMOVE", "FOR", "NAME", "OF", "THE", "AUTHORISED", "SIGNATORY", "SEAL"])
                                    
    return row


def handle_multi_line_fields(row, index1=None):
    if index1:
        row[index1] = remove_spl_char_multiple_pred(row[index1])
    return row


def handle_critical_fields(row, key, index1=None):
    if index1:
        row[index1] = remove_spl_char_multiple_pred(row[index1])
        if key =='lc_ref_no'or key == "lc_ref_number":
            regex = r"(?i)^(?:LC\s*NO\.?\s*|NO\.|no\.)"
            row[index1] = remove_label_name(row[index1],regex)
    return row

def handle_master_fields(row, key, index1=None):
    if key in amount_fields:
        _, row[index1] = extract_currency_and_amount(row[index1])
    if key in currency_fields:
        row[index1], _ = extract_currency_and_amount(row[index1])
    return row
    


look_loading = LookUp(config_['LookUp']['incoterms'], config_['LookUp']['countries'])
    
    
def generate_accuracy_data(occurrences, file, data, labels, predicted, label_category, required_keys_ = None):
    if required_keys_:
        keys_list = required_keys_
    else:
        keys_list = list(occurrences.keys())

    print(label_category.address_fields) 
    ratio_ = 100  #default ratio
    for key in keys_list:
        row = []
        row.append(file[0:-11] + ".png")
        # data = pre_handle_in_ic(key, labels, predicted, row, data)
        if key not in label_category.remove_fields:
            if key in labels and key in predicted:
                if len(labels[key]) == 1 and len(predicted[key]) == 1:
                    row.append(key)
                    row.append(str(labels[key][0][0]))
                    row.append(str(predicted[key][0][0]))
                    print("before pp >>>>>>>", row)
                    if key in label_category.numeric_fields:#['net_weight', 'gross_weight','total_quantity_of_goods']
                        ratio_ = config_['FuzzyRatio']['numeric_fields']
                        row = handle_numeric_fields(row,key, index1=2)
                        row = handle_numeric_fields(row, key, index1=3)
                    if key in label_category.single_word_fields:
                        ratio_ = config_['FuzzyRatio']['single_word_fields']
                        row = handle_single_word_fields(key, row,look_loading.incoterm_list, index1=2)
                        row = handle_single_word_fields(key, row,look_loading.incoterm_list, index1=3)
                
                    if key in label_category.address_fields: 
                        ratio_ = config_['FuzzyRatio']['address_fields']
                        row = handle_address_fields(row, index1=2)
                        row = handle_address_fields(row, index1=3)
                        
                    if key in label_category.single_line_fields: 
                        ratio_ = config_['FuzzyRatio']['single_line_fields']
                        row = handle_single_line_fields(key, row, index1=2)
                        row = handle_single_line_fields(key, row, index1=3)
                        
                    if key in label_category.date_fields: 
                        ratio_ = config_['FuzzyRatio']['date_fields']
                        row = handle_date_fields(row, index1=2)
                        row = handle_date_fields(row, index1=3)
                        
                    if key in label_category.multi_line_fields: 
                        ratio_ = config_['FuzzyRatio']['multi_line_fields']
                        row = handle_multi_line_fields(row, index1=2)
                        row = handle_multi_line_fields(row, index1=3)

                    if key in label_category.critical_fields: 
                        ratio_ = config_['FuzzyRatio']['critical_fields']
                        row = handle_critical_fields(row, key, index1=2)
                        row = handle_critical_fields(row, key, index1=3)
                        
                    if key in label_category.master_fields: 
                        ratio_ = config_['FuzzyRatio']['master_fields']
                        row = handle_master_fields(row, key, index1=2)
                        row = handle_master_fields(row, key, index1=3)
                    ratio_ = int(ratio_)
                    if key in label_category.numeric_fields:
                        try:
                            accuracy = fuzzy_float_comparison(float(str(row[2])), float(str(row[3])))
                            row.append(accuracy)
                            if accuracy >= ratio_:
                                row.append(1)
                            else:
                                row.append(0)
                
                        except:
                            row = get_fuzz_accuracy(row, ratio_)
                    else:
                        print('>>>>>>>>>>>>>>>>>>', row)
                        row = get_fuzz_accuracy(row, ratio_)
                            
                    row.append(predicted[key][0][2])
                    row.append(labels[key][0][1])
                    data.append(row)

                else:
                    # exit()
                    row.append(key)
                    actual = labels[key]
                    predicted_label = predicted[key]
                    print(f'second condition: {len(actual)}')
                    print(f'second condition: {len(predicted_label)}')
                    print(row)
                    l1 = len(actual)
                    l2 = len(predicted_label)
                    for i in range(l2):
                        if key in label_category.numeric_fields:
                            ratio_ = config_['FuzzyRatio']['numeric_fields']
                            predicted_label[i] = handle_numeric_fields(predicted_label[i],key, index1=0)
                            print(predicted_label)
                        if key in label_category.address_fields:
                            ratio_ = config_['FuzzyRatio']['single_word_fields']
                            predicted_label[i] = handle_address_fields(predicted_label[i], index1=0)
                        if key in label_category.single_word_fields:
                            ratio_ = config_['FuzzyRatio']['address_fields']
                            predicted_label[i] = handle_single_word_fields(key, predicted_label[i], look_loading.incoterm_list, index1=0)
                        if key in label_category.single_line_fields:
                            ratio_ = config_['FuzzyRatio']['single_line_fields']
                            predicted_label[i] = handle_single_line_fields(key, predicted_label[i], index1=0)
                            
                        if key in label_category.date_fields:
                            ratio_ = config_['FuzzyRatio']['date_fields']
                            predicted_label[i] = handle_date_fields(predicted_label[i], index1=0)
                        if key in label_category.critical_fields: 
                            ratio_ = config_['FuzzyRatio']['critical_fields']
                            predicted_label[i] = handle_critical_fields(predicted_label[i], key, index1=0)
                        if key in label_category.multi_line_fields: 
                            ratio_ = config_['FuzzyRatio']['multi_line_fields']
                            predicted_label[i] = handle_multi_line_fields(predicted_label[i], index1=0)
                            
                        if key in label_category.master_fields: 
                            ratio_ = config_['FuzzyRatio']['master_fields']
                            predicted_label[i] = handle_master_fields(predicted_label[i], key, index1=0)
                            
                            
                    for i in  range(l1):
                        if key in label_category.numeric_fields:
                            actual[i] = handle_numeric_fields(actual[i], key, index1=0)
                        if key in label_category.address_fields:
                            actual[i] = handle_address_fields(actual[i], index1=0)
                        if key in label_category.single_word_fields:
                            actual[i] = handle_single_word_fields(key, actual[i], look_loading.incoterm_list, index1=0)
                            
                        if key in label_category.single_line_fields:
                            actual[i] = handle_single_line_fields(key, actual[i], index1=0)
                            
                        if key in label_category.date_fields:
                            actual[i] = handle_date_fields(actual[i], index1=0)
                        if key in label_category.critical_fields: 
                            actual[i] = handle_critical_fields(actual[i], key, index1=0)

                        if key in label_category.multi_line_fields: 
                            actual[i] = handle_multi_line_fields(actual[i], index1=0)

                        if key in label_category.master_fields: 
                            actual[i] = handle_master_fields(actual[i], key, index1=0)
                           
                    print(f'after PP condition: {actual}')
                    print(f'after PP condition: {predicted_label}')
                    
                    ratio_ = int(ratio_)
                    to_do = [*range(0, l2, 1)]
                    print("starting to_do are", to_do)
                    for i in range(l1):
                        new_row = row.copy()
                        actual_value = actual[i]
                        print(f'the actual value is: {actual_value}')
                        print(f'the predicted value is: {predicted_label}')
                        to_do = filter_prediction(new_row, actual_value, predicted_label, to_do, data, ratio_)
                    print(to_do)
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
                    data.append(row)
                else:
                    predicted_label = predicted[key]
                    pre_val = '' 
                    for j in range(len(predicted_label)):
                        pre_val += predicted_label[j][0]+" "
                    #for val in predicted[key]:
                    new_row = row.copy()
                    new_row.append(pre_val)
                    new_row.append(0)
                    new_row.append(0)
                    new_row.append('')
                    new_row.append('')
                    data.append(new_row)
            else:
                continue
        
    return data


from datetime import datetime

def prepare_report(filtered_data, folder_path, doc_code_, category_name):
    
    column_names = []
    column_names.append("File_Name")
    column_names.append("label_name")
    column_names.append("actual")
    column_names.append("predicted")
    column_names.append("Accuracy")
    column_names.append("Match/No_Match")
    column_names.append("model_confidence")
    column_names.append("bbox")
    
    df = pd.DataFrame(filtered_data, columns=column_names)

    df.loc[df["Accuracy"].apply(float) >= 90, "Match/No_Match"] = 1

    res_path= os.path.join(folder_path, 'result_path')
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    os.makedirs(os.path.join(res_path, category_name), exist_ok=True)
    res_path = os.path.join(res_path, category_name)
    df.to_csv(f'{res_path}/{doc_code_}_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv')  
    
    # folder_name_path = glob.glob(f"{folder_path}/result_path/*")
    # folder_name_path.sort(reverse=True)
    # folder_name_path = folder_name_path[0]
    # print(folder_name_path)
    # # exit("+++++++++++++")

    # accuracy_generation_file = glob.glob(f"{folder_name_path}/*.csv")
    # accuracy_generation_file.sort(reverse=True)
    # print(f"Number of folders: {len(accuracy_generation_file)}")
    # # assert len(accuracy_generation_file) == 1	

    # accuracy_generation_file = accuracy_generation_file[0]

    accuracy_generation_file = os.path.join(folder_path, 'result_path', category_name, f'{doc_code_}_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv')
    file_path_csv1 = os.path.join(folder_path, 'result_path', category_name, f"{doc_code_}_{datetime.now().date()}_{datetime.now().hour}",
                            accuracy_generation_file)
    file_paths_csv2 = final_report(file_path_csv1, folder_path, doc_code_, category_name)

    # txt file generation
    stp_report(file_path_csv1, folder_path, doc_code_, category_name)

    final_overall_analysis(file_paths_csv2, file_path_csv1, folder_path, doc_code_, category_name)

