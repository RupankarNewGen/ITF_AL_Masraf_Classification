import numpy as np
import pandas as pd  # Assuming data is in CSV format, adjust accordingly for other formats
from post_process import load_data, handle_top_bottom_values, pre_handling_overlapping,\
    generate_accuracy_data, prepare_report
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
        

    # @staticmethod
    @classmethod
    def read_file(cls):
        cls.data_files, cls.result_files = load_data(result_path, data_path)
        return cls.data_files, cls.result_files
    
        # Implement file reading logic (assumes CSV for simplicity)
        # data = pd.read_csv(file_path)
        # return np.array(data.iloc[:, 0])  # Assuming a single column in the CSV file
        
    @classmethod
    def calculate_metrics(cls, required_keys_ = None):  
        occurrences: dict = {}
        data: list = []
        data_files, result_files = cls.read_file()
        
        for count, file in enumerate(data_files):
            print("count is:", count)
            with open(os.path.join(data_path, file), "r") as f:
                labels = json.load(f)
                try:
                    labels = list(labels)
                except:
                    continue

            # count of the labels in each file
            for label in labels:
                if label in occurrences:
                    occurrences[label] += 1
                else:
                    occurrences[label] = 1        
        for file, predicted_files in zip(data_files, result_files):
            try:
                with open(os.path.join(data_path, file), "r") as f:
                    labels = json.load(f)
            except Exception as e:
                print(f"Exception is {e}")
                print(f"Error opening a data file named :{data_path}")
            try:
                print(os.path.join(result_path, file[0:-11] + "1.txt"))
                with open(os.path.join(result_path, file[0:-11] + "1.txt"), "r") as f2:
                    predicted = json.load(f2)
            except IOError as _:
                if debug_mode:
                    print("some problem opening file")
                # logger.info(f"some problem opening file named: {result_path}{file[0:-11]}1.txt")

                try:
                    with open(os.path.join(result_path, file[0:-11] + "_s_11.txt"), "r") as f2:
                        predicted = json.load(f2)
                except IOError as _:
                    print("still not opened")
                    continue

            print(f'actual labels: {labels}')
            print(f'number of keys : {len(list(labels.keys()))}')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++=')
            print(f'predicted labels: {predicted}')
            print(f'predicted labels: {len(predicted)}')

            if predicted == {} and labels == {}:
                print("*******")
                continue
            #mapping top and bottom values 
            
            labels, predicted = handle_top_bottom_values(labels, predicted, cls.doc_code_)
            labels, predicted = pre_handling_overlapping(labels, predicted, cls.doc_code_)
            print(list(occurrences.keys()))
            data = generate_accuracy_data(occurrences, file, data, labels, predicted, cls.label_category, required_keys_ = required_keys_)
            
        return data
        ''' Actual accuracy generation logic'''
    @classmethod
    def label_wise_report(cls):
        
        accuracy_data = cls.calculate_metrics()
        prepare_report(accuracy_data, folder_path, doc_code_, "label_wise")
        
        # Calculate and print label-wise metrics
        # ...
    @classmethod
    def rules_required_keys_report(cls):
        accuracy_data = cls.calculate_metrics(required_keys_= cls.rules_required_keys)
        prepare_report(accuracy_data, folder_path, doc_code_, "rules_required")
        
        # Calculate and print label-wise metrics
        # ...
    @classmethod
    def document_wise_report(cls):
        # Calculate and print document-wise metrics
        # ...
        pass
    @classmethod
    def most_occurred_label_report(cls):
        accuracy_data = cls.calculate_metrics(required_keys_= cls.most_frequently_occurring_keys)
        prepare_report(accuracy_data, folder_path, doc_code_, "most_frequently")
        # Identify and print the most occurred label
        # Calculate and print metrics for the most occurred label
        # ...
    @classmethod
    def layoutwise_report_generation(cls):
        
        # Calculate and print layout-wise metrics
        # Calculate and print metrics for layouts used during training
        # Calculate and print metrics for layouts not used during training
        # ...
        pass
    @classmethod
    def generate_reports(cls):
        cls.label_wise_report()
        cls.rules_required_keys_report()
        cls.most_occurred_label_report()
        




if __name__ == '__main__':

    product_config = ConfigParser()
    debug_mode = True
    # relative path => passed in validation
    product_config.read("post_processing/config/config.ini")
    prod_code = product_code_map[product_config["Product"]["code"]]
    doc_code = product_config["Product"]["document_code"]
    if '[' in doc_code:
        doc_elements = doc_code[1:-1].split(', ')
    # Convert elements to a Python list
    doc_code_list = [element.strip() for element in doc_elements]
    print(doc_code)
    #best_keys_list = ast.literal_eval(configur[f'{ground_truth}_BEST_KEYS']['keys'])
    # data folder path
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
        #******************* *************************************************************************
        #******************* ************************************************************************
        result_path = os.path.join(folder_path, "Results_Images")
        data_path = os.path.join(folder_path, "New_Master_Data_Merged")
        
        evaluator = InformationExtractionEvaluator(folder_path, result_path, data_path, pp_json, doc_code_.upper(), rules_keys[doc_code_.upper()], most_frequently_occurring_keys[doc_code_.upper()])
        
        evaluator.generate_reports()
 