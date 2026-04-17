import os
import glob
import json
import pandas as pd
from datetime import datetime
from configparser import ConfigParser
from fuzzywuzzy import fuzz
from label_category import LabelCategory

config_ = ConfigParser()
config_.read("post_processing/config.ini")
match_name = "Match/No_Match"

def load_data(result_path, data_path):
    data_files = os.listdir(data_path)
    result_files = os.listdir(result_path)

    new = []
    new_result = []

    for file in data_files:
        if str(file)[-10:] == "labels.txt":
            new.append(file)
            new_result.append(file[:-11] + ".txt")

    return new, new_result


def fuzzy_float_comparison(float1, float2, tolerance=1e-6):
    absolute_difference = abs(float1 - float2)
    if absolute_difference <= tolerance:
        return 100
    else:
        if max(abs(float1), abs(float2)) == 0:
            return 100
        similarity = 100 - (absolute_difference / max(abs(float1), abs(float2))) * 100
        return similarity


def get_fuzz_accuracy(row, fuzzy_ratio):
    fuzzy_accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
    row.append(fuzzy_accuracy)
    if fuzzy_accuracy >= fuzzy_ratio:
        row.append(1)
    else:
        row.append(0)
    return row


def generate_accuracy_data(occurrences, file, data, labels, predicted, label_category, required_keys_=None):
    if required_keys_:
        keys_list = required_keys_
    else:
        keys_list = list(occurrences.keys())

    ratio_ = 100
    for key in keys_list:
        row = []
        row.append(file[0:-11] + ".png")
        if key not in label_category.remove_fields:
            if key in labels and key in predicted:
                act_val = str(labels[key][0][0])
                try:
                    pred_val = str(predicted[key]["value"])
                except Exception as e:
                    pred_val = ""
                try:
                    pred_box = predicted[key]["coordinate"][0]
                except Exception as e:
                    pred_box = []
                act_box = labels[key][0][1]

                row.append(key)
                row.append(act_val)
                row.append(pred_val)

                # Determine ratio
                if key in label_category.numeric_fields:
                    ratio_ = config_['FuzzyRatio']['numeric_fields']
                elif key in label_category.single_word_fields:
                    ratio_ = config_['FuzzyRatio']['single_word_fields']
                elif key in label_category.address_fields: 
                    ratio_ = config_['FuzzyRatio']['address_fields']
                elif key in label_category.single_line_fields: 
                    ratio_ = config_['FuzzyRatio']['single_line_fields']
                elif key in label_category.date_fields: 
                    ratio_ = config_['FuzzyRatio']['date_fields']
                elif key in label_category.multi_line_fields: 
                    ratio_ = config_['FuzzyRatio']['multi_line_fields']
                elif key in label_category.critical_fields: 
                    ratio_ = config_['FuzzyRatio']['critical_fields']
                elif key in label_category.master_fields: 
                    ratio_ = config_['FuzzyRatio']['master_fields']
                
                ratio_ = int(ratio_)
                
                if key in label_category.numeric_fields:
                    try:
                        import re
                        act_f = float(re.sub(r'[^\d.]', '', act_val))
                        pred_f = float(re.sub(r'[^\d.]', '', pred_val))
                        accuracy = fuzzy_float_comparison(act_f, pred_f)
                        row.append(accuracy)
                        if accuracy >= ratio_:
                            row.append(1)
                        else:
                            row.append(0)
                    except:
                        row = get_fuzz_accuracy(row, ratio_)
                else:
                    row = get_fuzz_accuracy(row, ratio_)
                
                row.append(pred_box)
                row.append(act_box)
                data.append(row)

            elif key in labels and key not in predicted:
                row.append(key)
                row.append(str(labels[key][0][0]))
                row.append("")
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(labels[key][0][1])
                data.append(row)

            elif key not in labels and key in predicted:
                row.append(key)
                row.append("")
                try:
                    pred_val = str(predicted[key]["value"])
                except Exception as e:
                    pred_val = ""
                try:
                    pred_box = predicted[key]["coordinate"][0]
                except Exception as e:
                    pred_box = []

                row.append(pred_val)
                row.append(0)
                row.append(0)
                row.append(pred_box)
                row.append("")
                data.append(row)
            else:
                continue
    return data


def final_overall_analysis(path1, path2, folder_path, doc_code, category_name):
    accuracy_lookup = pd.read_csv(path1)
    analysis_report = pd.read_csv(path2)
    data = []
    label_names = []

    for index, row in accuracy_lookup.iterrows():
        if row["Complete_Match_Percentage"] >= 80 and row["Fuzzy_Match_Percentage"] >= 85 and row["Label_Name"] != "OVERALL":
            label_names.append(str(row["Label_Name"]))
            data.append(list(row)[1:])

    output_dir = os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")
    os.makedirs(output_dir, exist_ok=True)

    save_path_1 = os.path.join(output_dir, f"{doc_code}_Final_Report_Summary_Best_Fields.csv")
    pd.DataFrame(data, columns=list(accuracy_lookup.columns)[1:]).to_csv(save_path_1)

    new_data = [list(row)[1:] for index, row in analysis_report.iterrows() if str(row["label_name"]) in label_names]
    save_path_2 = os.path.join(output_dir, f"{doc_code}_Overall_Analysis_Best_Fields.csv")
    pd.DataFrame(new_data, columns=list(analysis_report.columns)[1:]).to_csv(save_path_2)


def final_report(csv01, folder_path, doc_code, category_name):
    df = pd.read_csv(csv01)
    label_names, label_counts, label_detected, detection_accuracy, average_accuracy, matched_labels, matched_detected, total_match_percentage = [], [], [], [], [], [], [], []
    g = df.groupby("label_name")
    
    for name, name_df in g:
        label_names.append(name)
        label_counts.append(len(name_df.index))
        average_accuracy.append(round(name_df["Accuracy"].mean(), 2))
        matched = sum(list(name_df[match_name]))
        matched_labels.append(matched)
        total_match_percentage.append(round((matched / len(name_df.index)) * 100, 2))
        not_detected = name_df["predicted"].isnull().sum()
        detected = len(name_df.index) - not_detected
        label_detected.append(detected)
        
        if detected > 0:
            matched_detected.append(matched / detected)
        else:
            matched_detected.append(0)
            
        if len(name_df.index) > 0:
            detection_accuracy.append((detected / len(name_df.index)) * 100)
        else:
            detection_accuracy.append(0)

    data = {
        'Label_Name': label_names, 'Label_Count': label_counts, "Labels_Detected": label_detected,
        "Detection_Accuracy": detection_accuracy, "Fuzzy_Match_Percentage": average_accuracy,
        "Complete_Match_Count_Detected": matched_detected, "Complete_Match_Count": matched_labels,
        "Complete_Match_Percentage": total_match_percentage
    }

    detected = sum(label_detected)
    all_matched = sum(matched_labels)
    avg_matched_detected = all_matched / detected if detected > 0 else 0
    all_labels = sum(label_counts)
    overall_detection_accuracy = (detected / all_labels) * 100 if all_labels > 0 else 0

    report = pd.DataFrame(data)
    li = ["OVERALL", all_labels, detected, overall_detection_accuracy,
          df["Accuracy"].mean(), avg_matched_detected, all_matched,
          all_matched / all_labels if all_labels > 0 else 0]
    report.loc[len(report.index)] = li

    output_dir = os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")
    os.makedirs(output_dir, exist_ok=True)

    file_path_csv2 = os.path.join(output_dir, f"final_report_{doc_code}.csv")
    report.to_csv(file_path_csv2)

    name_path = os.path.join(output_dir, f"final_report_{datetime.now()}.txt")
    text_report = {
        row["Label_Name"]: {
            "Label_Count": row["Label_Count"],
            "Detection_Accuracy": row["Detection_Accuracy"],
            "Fuzzy_Match_Percentage": row["Fuzzy_Match_Percentage"],
            "Complete_Match_Count_Detected": row["Complete_Match_Count_Detected"],
            "Complete_Match_Count": row["Complete_Match_Count"],
            "Complete_Match_Percentage": row["Complete_Match_Percentage"],
        } for index, row in report.iterrows()
    }
    
    with open(name_path, "w") as f:
        json.dump(text_report, f)
    
    return file_path_csv2


def stp_report(csv02, folder_path, doc_code, category_name):
    df = pd.read_csv(csv02)
    g = df.groupby("File_Name")
    stp_data, num_files, match_files = [], 0, 0

    for name, name_df in g:
        num_files += 1
        match_value = sum(list(name_df[match_name]))
        num_labels = len(name_df[match_name])
        flag = 1 if match_value == num_labels else 0
        if flag == 1:
            match_files += 1
        stp_data.append([str(name), match_value, num_labels, flag])

    df2 = pd.DataFrame(stp_data, columns=["File_Name", "Labels_Matched", "Labels_Present", "STP_Match"])
    
    output_dir = os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")
    os.makedirs(output_dir, exist_ok=True)
    df2.to_csv(os.path.join(output_dir, f"STP_Summary_{datetime.now()}.csv"))


def prepare_report(filtered_data, folder_path, doc_code_, category_name):
    column_names = ["File_Name", "label_name", "actual", "predicted", "Accuracy", "Match/No_Match", "bbox", "bbox_ground_truth"]
    df = pd.DataFrame(filtered_data, columns=column_names)
    df.loc[df["Accuracy"].apply(float) >= 90, "Match/No_Match"] = 1

    res_path = os.path.join(folder_path, 'result_path', category_name)
    os.makedirs(res_path, exist_ok=True)
    df.to_csv(os.path.join(res_path, f'{doc_code_}_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv'))  
    
    accuracy_generation_file = os.path.join(res_path, f'{doc_code_}_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv')
    file_paths_csv2 = final_report(accuracy_generation_file, folder_path, doc_code_, category_name)
    stp_report(accuracy_generation_file, folder_path, doc_code_, category_name)
    final_overall_analysis(file_paths_csv2, accuracy_generation_file, folder_path, doc_code_, category_name)
