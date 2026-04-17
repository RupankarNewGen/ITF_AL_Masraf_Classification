import numpy as np
import pandas as pd
from post_process_rupankar import load_data, generate_accuracy_data, prepare_report
from configparser import ConfigParser
from config.prod_mapping import product_code_map, document_code_map 
from label_category import LabelCategory
import json
import os


class InformationExtractionEvaluator:
    @classmethod
    def __init__(cls, folder_path, result_path, data_path, pp_json, doc_code_, rules_required_keys, most_frequently_occurring_keys):
        cls.file_path = folder_path
        cls.result_path = result_path
        cls.data_path = data_path
        cls.doc_code_ = doc_code_
        cls.rules_required_keys = rules_required_keys
        cls.most_frequently_occurring_keys = most_frequently_occurring_keys
        cls.label_category = LabelCategory() 
        
        with open(pp_json, 'r') as file:
            pl_keys = json.load(file)
        for method, fields in pl_keys.items():
            for field in fields:
                getattr(cls.label_category, f'add_{method}')(field)
        
        cls.label_category.display_category_summary()

    @classmethod
    def read_file(cls):
        cls.data_files, cls.result_files = load_data(cls.result_path, cls.data_path)
        return cls.data_files, cls.result_files
        
    @classmethod
    def calculate_metrics(cls, required_keys_=None):  
        occurrences: dict = {}
        data: list = []
        data_files, result_files = cls.read_file()
        
        for count, file in enumerate(data_files):
            print("count is:", count)
            with open(os.path.join(cls.data_path, file), "r") as f:
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
                    
        for file in data_files:
            try:
                with open(os.path.join(cls.data_path, file), "r") as f:
                    labels = json.load(f)
            except Exception as e:
                print(f"Exception is {e}")
                continue
                
            predicted_raw = {}
            pred_path_json = os.path.join(cls.result_path, file[0:-11] + ".json")
            pred_path_txt = os.path.join(cls.result_path, file[0:-11] + "1.txt")
            
            try:
                if os.path.exists(pred_path_json):
                    with open(pred_path_json, "r") as f2:
                        predicted_raw = json.load(f2)
                elif os.path.exists(pred_path_txt):
                    with open(pred_path_txt, "r") as f2:
                        predicted_raw = json.load(f2)
            except IOError as _:
                pass

            if not predicted_raw:
                print(f"Could not load prediction for {file}")
                # We evaluate even an empty dict so it logs 0% match instead of crashing
                predicted = {}
                predicted_raw = {"dummy": {}}
                image_key = "dummy"
            else:
                image_key = list(predicted_raw.keys())[0]
                if "extraction_result" in predicted_raw[image_key]:
                    predicted = predicted_raw[image_key]["extraction_result"]
                else:
                    predicted = {}

            # Retrieve signature and stamp
            if "docAutoSignDetection" in predicted_raw[image_key]:
                sign_bboxes = predicted_raw[image_key]["docAutoSignDetection"].get("keys_bboxes", {}).get("sign", [])
                if sign_bboxes:
                    predicted["sign"] = {"value": "", "coordinate": sign_bboxes}
                    
            if "docAutoStampDetection" in predicted_raw[image_key]:
                stamp_bboxes = predicted_raw[image_key]["docAutoStampDetection"].get("keys_bboxes", {}).get("stamp", [])
                if stamp_bboxes:
                    predicted["stamp"] = {"value": "", "coordinate": stamp_bboxes}

            # Dropped overlapping handlers since data is explicitly single-value now
            data = generate_accuracy_data(occurrences, file, data, labels, predicted, cls.label_category, required_keys_=required_keys_)
            
        return data

    @classmethod
    def label_wise_report(cls):
        accuracy_data = cls.calculate_metrics()
        prepare_report(accuracy_data, cls.file_path, cls.doc_code_, "label_wise")
        
    @classmethod
    def rules_required_keys_report(cls):
        accuracy_data = cls.calculate_metrics(required_keys_= cls.rules_required_keys)
        prepare_report(accuracy_data, cls.file_path, cls.doc_code_, "rules_required")
        
    @classmethod
    def most_occurred_label_report(cls):
        accuracy_data = cls.calculate_metrics(required_keys_= cls.most_frequently_occurring_keys)
        prepare_report(accuracy_data, cls.file_path, cls.doc_code_, "most_frequently")

    @classmethod
    def generate_reports(cls):
        cls.label_wise_report()
        cls.rules_required_keys_report()
        cls.most_occurred_label_report()


if __name__ == '__main__':
    product_config = ConfigParser()
    debug_mode = True
    product_config.read("post_processing/config/config.ini")
    prod_code = product_code_map[product_config["Product"]["code"]]
    doc_code = product_config["Product"]["document_code"]
    if '[' in doc_code:
        doc_elements = doc_code[1:-1].split(', ')
    doc_code_list = [element.strip() for element in doc_elements]
    print(doc_code)
    
    product_wise_folder = ConfigParser()
    product_wise_folder.read("post_processing/config/prod.ini")
    post_processing_config = ConfigParser()
    post_processing_config.read("post_processing/config.ini")
    rules_keys_json = post_processing_config['PATH']['rules_required_keys']
    most_frequently_occurring_keys_json = post_processing_config['PATH']['most_frequently_occurring_keys']
    
    with open(rules_keys_json, 'r') as file:
        rules_keys = json.load(file)
        
    with open(most_frequently_occurring_keys_json, 'r') as file:
        most_frequently_occurring_keys = json.load(file)
        
    for doc_code_ in doc_code_list:
        pp_json = post_processing_config[prod_code][doc_code_]
        doc_code = document_code_map[doc_code_]
        folder_path = product_wise_folder[prod_code][doc_code]

        result_path = os.path.join(folder_path, "Results_Images")
        data_path = os.path.join(folder_path, "New_Master_Data_Merged")
        
        evaluator = InformationExtractionEvaluator(folder_path, result_path, data_path, pp_json, doc_code_.upper(), rules_keys[doc_code_.upper()], most_frequently_occurring_keys[doc_code_.upper()])
        
        evaluator.generate_reports()