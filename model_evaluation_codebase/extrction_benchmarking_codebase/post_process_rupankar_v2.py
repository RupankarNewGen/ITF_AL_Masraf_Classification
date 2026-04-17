import os
import glob
import json
import re
import pandas as pd
from datetime import datetime
from configparser import ConfigParser
from fuzzywuzzy import fuzz
from label_category import LabelCategory

# --- NEW IMPORTS ---
from utility import *
from pre_mapping import PredictionKeyMapping, ParentKeyMapping, OverLappingKeys, \
    transport_fields, weights_fields, amount_fields, currency_fields, \
        bottom_value, top_value

config_ = ConfigParser()
config_.read("config.ini")
match_name = "Match/No_Match"

class LookUp:
    def __init__(self, incoterm_path, countries_path) -> None:
        self.incoterm_path = incoterm_path
        self.countries_path = countries_path
        self.incoterm_list = []
        try:
            with open(self.incoterm_path, 'r') as file:
                for line in file:
                     self.incoterm_list.append(line.strip())
        except Exception:
            pass
        
        self.countries_list = []
        try:
            with open(self.countries_path, 'r') as file:
                for line in file:
                     self.countries_list.append(line.strip())
        except Exception:
            pass

look_loading = LookUp(config_['LookUp']['incoterms'], config_['LookUp']['countries'])

# Post-processing helper functions
def clean_numeric_field(val: str, key: str) -> str:
    if val is None: return ""
    try:
        val = remove_spl_char_multiple_pred(str(val))
        val = remove_spaces(val)
        if key in weights_fields:
            val = keep_specific_characters(val)
        else:
            val = remove_alphabets(val)
    except Exception:
        pass
    return str(val)

def clean_address_field(val: str) -> str:
    if val is None: return ""
    try:
        val = str(val)
        val = remove_symbols(val)
        val = remove_spl_char_multiple_pred(val)
        val = remove_text_after_phrases(filter_address(val))
        if val is not None:
            val = remove_spaces(val)
    except Exception:
        pass
    return str(val)

def clean_single_word_field(val: str, key: str, incoterm_list: list) -> str:
    if val is None: return ""
    try:
        val = str(val)
        if key == "incoterm":
            val = preprocess_incoterm(val, incoterm_list)
        if key in transport_fields:
            val = remove_words(val, ['BY',"EXPORT", 'freight'])
            val = remove_by_prefix(val)
        if key == 'csh_drawn_under_rules':
            val = pp_cash_drawn_rules(val)
        if key == 'port_of_discharge':
            regex="([PDo])?(o)?(r)?(t)?(\s)?(o)?(f)?(\s)?([DO0])?(i)?(s)?(c)?(h)?(a)?(r)?(g)?(e)?"
            val = remove_label_name(val, regex)
        if key == 'port_of_loading':
            regex=r"([PDo])?(o)?(r)?(t)?(\s)?(o)?(f)?(\s)?([IL1T\|])?(o)?(a)?(d)?(i)?(n)?(g)?"
            val = remove_label_name(val, regex)
        if key == 'page_no' or key == "original_Number":
            regex = r"(?i)(page|no\.|no|pago\s*no\.|pago|Page)"
            val = generic_page_no(remove_special_chars(remove_label_name(val, regex)))
        if key == 'to_place':
            regex = r"(?i)^To\s*:"
            val = remove_label_name(val, regex)
        if key == "from_place":
            val = remove_words(val, ['FROM'])
    except Exception:
        pass
    return str(val)

def clean_single_line_field(val: str, key: str) -> str:
    if val is None: return ""
    try:
        val = remove_spl_char_multiple_pred(str(val))
        if key == "doc_delivery_instruction":
            val = doc_delivery_pp(val)
        if key == "doc_charge_instructions":
            val = doc_chages_pp(val) 
        if key == 'consignee_name':
            regex = r"(?i)NOTIFY\s*:\s*"
            val = remove_label_name(val, regex)
        if key == "consignor_name":
            val = remove_words(val, ['EXPORTER'])
        if key == "declaration_by" or key == "declaration":
            val = remove_words(val, ["REMOVE", "FOR", "NAME", "OF", "THE", "AUTHORISED", "SIGNATORY", "SEAL"])
    except Exception:
        pass
    return str(val)

def clean_date_field(val: str) -> str:
    if val is None: return ""
    try:
        val = clean_date(str(val))
    except Exception:
        pass
    return str(val)

def clean_multi_line_field(val: str) -> str:
    if val is None: return ""
    try:
        val = remove_spl_char_multiple_pred(str(val))
    except Exception:
        pass
    return str(val)

def clean_critical_field(val: str, key: str) -> str:
    if val is None: return ""
    try:
        val = remove_spl_char_multiple_pred(str(val))
        if key == 'lc_ref_no' or key == "lc_ref_number":
            regex = r"(?i)^(?:LC\s*NO\.?\s*|NO\.|no\.)"
            val = remove_label_name(val, regex)
    except Exception:
        pass
    return str(val)

def clean_master_field(val: str, key: str) -> str:
    if val is None: return ""
    try:
        val = str(val)
        if key in amount_fields:
            _, val = extract_currency_and_amount(val)
        if key in currency_fields:
            val, _ = extract_currency_and_amount(val)
    except Exception:
        pass
    return str(val)


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


def _get_ratio_and_clean_pred(key, pred_val_raw, label_category):
    """Determine the fuzzy match ratio and clean the prediction value for a given field key.
    Returns (ratio_int, cleaned_pred_str).
    Called ONCE per key (prediction is always a single value in the current API format).
    """
    ratio_ = 100
    pred_val = pred_val_raw
    if key in label_category.numeric_fields:
        ratio_ = int(config_['FuzzyRatio']['numeric_fields'])
        pred_val = clean_numeric_field(pred_val_raw, key)
    elif key in label_category.single_word_fields:
        ratio_ = int(config_['FuzzyRatio']['single_word_fields'])
        pred_val = clean_single_word_field(pred_val_raw, key, look_loading.incoterm_list)
    elif key in label_category.address_fields:
        ratio_ = int(config_['FuzzyRatio']['address_fields'])
        pred_val = clean_address_field(pred_val_raw)
    elif key in label_category.single_line_fields:
        ratio_ = int(config_['FuzzyRatio']['single_line_fields'])
        pred_val = clean_single_line_field(pred_val_raw, key)
    elif key in label_category.date_fields:
        ratio_ = int(config_['FuzzyRatio']['date_fields'])
        pred_val = clean_date_field(pred_val_raw)
    elif key in label_category.multi_line_fields:
        ratio_ = int(config_['FuzzyRatio']['multi_line_fields'])
        pred_val = clean_multi_line_field(pred_val_raw)
    elif key in label_category.critical_fields:
        ratio_ = int(config_['FuzzyRatio']['critical_fields'])
        pred_val = clean_critical_field(pred_val_raw, key)
    elif key in label_category.master_fields:
        ratio_ = int(config_['FuzzyRatio']['master_fields'])
        pred_val = clean_master_field(pred_val_raw, key)
    return ratio_, pred_val


def _clean_act_val(key, act_val_raw, label_category):
    """Apply the same category-specific cleaning to a single GT string.
    Called once per GT entry inside the GT loop.
    """
    if key in label_category.numeric_fields:
        return clean_numeric_field(act_val_raw, key)
    elif key in label_category.single_word_fields:
        return clean_single_word_field(act_val_raw, key, look_loading.incoterm_list)
    elif key in label_category.address_fields:
        return clean_address_field(act_val_raw)
    elif key in label_category.single_line_fields:
        return clean_single_line_field(act_val_raw, key)
    elif key in label_category.date_fields:
        return clean_date_field(act_val_raw)
    elif key in label_category.multi_line_fields:
        return clean_multi_line_field(act_val_raw)
    elif key in label_category.critical_fields:
        return clean_critical_field(act_val_raw, key)
    elif key in label_category.master_fields:
        return clean_master_field(act_val_raw, key)
    return act_val_raw


def _compute_accuracy_row(row, key, act_val, pred_val, ratio_, pred_box, act_box, label_category):
    """Append accuracy, match flag, and bbox columns to row and return it."""
    accuracy, match_flag = _compute_score_only(key, act_val, pred_val, ratio_, label_category)
    row.append(accuracy)
    row.append(match_flag)
    row.append(pred_box)
    row.append(act_box)
    return row


def _compute_score_only(key, act_val, pred_val, ratio_, label_category):
    """Compute (accuracy_score, match_flag) without building a row.
    Used during greedy alignment to test candidate prediction parts before committing.
    """
    if key in label_category.numeric_fields:
        try:
            act_f = float(re.sub(r'[^\d.]', '', act_val))
            pred_f = float(re.sub(r'[^\d.]', '', pred_val))
            accuracy = fuzzy_float_comparison(act_f, pred_f)
        except Exception:
            accuracy = fuzz.ratio(str(act_val).lower(), str(pred_val).lower())
    else:
        accuracy = fuzz.ratio(str(act_val).lower(), str(pred_val).lower())
    return accuracy, (1 if accuracy >= ratio_ else 0)


def generate_accuracy_data(occurrences, file, data, labels, predicted, label_category, required_keys_=None):
    """
    Compare GT labels against model predictions and build the per-field accuracy data list.

    Multi-value GT + Multi-value Prediction support (v2 upgrade):

        GT side (yolo_to_gt_converter):
            A field key can have multiple [value, bbox] entries, one per annotated bbox.

        Prediction side (API format):
            The model joins multiple occurrences of the same field with a "~" separator
            in the "value" string and lists corresponding bboxes in "coordinate".
            e.g. {"value": "TEXT1~TEXT2", "coordinate": [[bbox1], [bbox2]]}

        Strategy — Greedy 1-to-1 alignment (inspired by original post_process.py):
          1. Split prediction on "~" → pred_parts list; align with coordinate list.
          2. Clean every pred_part individually (same category rules).
          3. For each GT entry, scan through unmatched pred_parts:
               - If fuzzy score >= ratio_ → MATCH: write matched row, mark pred as used.
               - If no pred matches → NO-MATCH: write empty-prediction row (score=0).
          4. One CSV row is written per GT entry.
          5. Single-value case (no "~") degrades gracefully to simple 1-to-1.
    """
    if required_keys_:
        keys_list = required_keys_
    else:
        keys_list = list(occurrences.keys())

    file_prefix = file[0:-11] + ".png"

    for key in keys_list:
        if key in label_category.remove_fields:
            continue

        # ------------------------------------------------------------------ #
        # CASE 1 — GT present AND prediction present                          #
        # Handles: single value, multi-value merged with ~, multi-bbox coords #
        # ------------------------------------------------------------------ #
        if key in labels and key in predicted:

            # --- Read raw prediction value and coordinate list ---
            try:
                pred_val_raw = str(predicted[key]["value"])
            except Exception:
                pred_val_raw = ""
            pred_coords = predicted[key].get("coordinate", []) if isinstance(predicted[key], dict) else []

            # --- Split on ~ to get individual prediction parts ---
            pred_parts_raw = [p.strip() for p in pred_val_raw.split("~")]

            # --- Determine ratio (same for all parts of the same field key) ---
            ratio_, _ = _get_ratio_and_clean_pred(key, "", label_category)

            # --- Clean each prediction part independently ---
            pred_parts_clean = [_get_ratio_and_clean_pred(key, p, label_category)[1]
                                 for p in pred_parts_raw]

            # --- Map each prediction part to its bbox (pad with [] if fewer coords) ---
            pred_boxes = [pred_coords[i] if i < len(pred_coords) else []
                          for i in range(len(pred_parts_raw))]

            # --- Greedy alignment: each GT entry consumes at most one pred_part ---
            to_do = list(range(len(pred_parts_clean)))  # unmatched prediction indices

            for gt_entry in labels[key]:
                try:
                    act_val_raw = str(gt_entry[0])
                except Exception:
                    act_val_raw = ""
                try:
                    act_box = gt_entry[1]
                except Exception:
                    act_box = []

                act_val = _clean_act_val(key, act_val_raw, label_category)

                matched = False
                for j in list(to_do):          # iterate over a copy so we can remove
                    score, flag = _compute_score_only(
                        key, act_val, pred_parts_clean[j], ratio_, label_category)
                    if flag == 1:               # score >= ratio_ → best first match
                        row = [file_prefix, key, act_val, pred_parts_clean[j],
                               score, 1, pred_boxes[j], act_box]
                        data.append(row)
                        to_do.remove(j)         # mark this pred_part as consumed
                        matched = True
                        break

                if not matched:
                    # No prediction part cleared the threshold for this GT entry
                    # Still write a row showing the best available score (informative)
                    if to_do:
                        # Compare against each remaining pred and take best score
                        best_j = max(to_do,
                                     key=lambda j: _compute_score_only(
                                         key, act_val, pred_parts_clean[j],
                                         ratio_, label_category)[0])
                        best_score, _ = _compute_score_only(
                            key, act_val, pred_parts_clean[best_j], ratio_, label_category)
                        row = [file_prefix, key, act_val, pred_parts_clean[best_j],
                               best_score, 0, pred_boxes[best_j], act_box]
                    else:
                        # All predictions already consumed by earlier GT entries
                        row = [file_prefix, key, act_val, "", 0, 0, [], act_box]
                    data.append(row)

        # ------------------------------------------------------------------ #
        # CASE 2 — GT present, no prediction (model missed the field)         #
        # ------------------------------------------------------------------ #
        elif key in labels and key not in predicted:
            # Loop over all GT entries so every missed annotation is recorded
            for gt_entry in labels[key]:
                try:
                    act_val = str(gt_entry[0])
                except Exception:
                    act_val = ""
                try:
                    act_box = gt_entry[1]
                except Exception:
                    act_box = []

                row = [file_prefix, key, act_val, "", 0, 0, [], act_box]
                data.append(row)

        # ------------------------------------------------------------------ #
        # CASE 3 — Prediction present, no GT (model hallucinated a field)     #
        # ------------------------------------------------------------------ #
        elif key not in labels and key in predicted:
            try:
                pred_val = str(predicted[key]["value"])
            except Exception:
                pred_val = ""
            try:
                pred_box = predicted[key]["coordinate"][0]
            except Exception:
                pred_box = []

            row = [file_prefix, key, "", pred_val, 0, 0, pred_box, ""]
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
    
    # --- Generate Additional Parallel 75% Target Threshold CSV ---
    df_75 = pd.DataFrame(filtered_data, columns=column_names)
    df_75["Match/No_Match"] = df_75["Accuracy"].apply(lambda x: 1 if float(x) >= 75.0 else 0)
    df_75.to_csv(os.path.join(res_path, f'{doc_code_}_analysis_fuzzy_75.csv'), index=False)
    
    accuracy_generation_file = os.path.join(res_path, f'{doc_code_}_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv')
    file_paths_csv2 = final_report(accuracy_generation_file, folder_path, doc_code_, category_name)
    stp_report(accuracy_generation_file, folder_path, doc_code_, category_name)
    final_overall_analysis(file_paths_csv2, accuracy_generation_file, folder_path, doc_code_, category_name)
