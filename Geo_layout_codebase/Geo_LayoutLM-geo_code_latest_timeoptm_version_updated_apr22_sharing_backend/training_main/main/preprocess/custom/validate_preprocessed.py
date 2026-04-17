"""
Run this script incase you want to observe the keys missed/dropped during data preparation.
"""

import os
import json

actual_labels_path = '/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Layoutlmv3_code_base/BOL_complete_data/Labels'
# prepared_data_path = '/home/gayathri/g3/extraction/GeoLayoutLM/preprocess/custom/custom_data/data_in_funsd_format/testing_data/annotations/'

prepared_data_path = '/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Layoutlmv3_code_base/BOL_complete_data/data_in_funsd_format/dataset/custom_geo/preprocessed'
classes_path = '/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Layoutlmv3_code_base/BOL_complete_data/label.txt'

with open(classes_path, 'r') as f:
    classes = f.readlines()

classes = [item.replace('\n', '').strip() for item in classes]
class_map = {i: item for i, item in enumerate(classes)}

class_counts = {class_map[item] : 0 for item in class_map}


# for item in class_map:
#     class_counts[class_map[item]] = 

actual_labels_list = os.listdir(actual_labels_path)
prepared_data_list = os.listdir(prepared_data_path)


for labels in actual_labels_list:
    # with open(os.path.join(actual_labels_path, labels), 'r') as f:
    #         actual_labels = f.readlines()
    # actual_labels = [class_map[int(item.split()[0])] for item in actual_labels if int(item.split()[0]) < len(class_map)]
    # for lbl in actual_labels:
    #      class_counts[lbl] += 1

    if labels.replace('txt', 'json') in prepared_data_list:
        # actual_classes = 
        with open(os.path.join(actual_labels_path, labels), 'r') as f:
            actual_labels = f.readlines()

        actual_labels = [class_map[int(item.split()[0])] for item in actual_labels]

        with open(os.path.join(prepared_data_path, labels.replace('txt', 'json')), 'r') as f:
            prepared_data = json.load(f)


        prepared_labels = [item for item in prepared_data['parse']['class'] 
                           if len(prepared_data['parse']['class'][item])!=0 and item!='O']

        # prepared_labels = [item['label'] for item in prepared_data['form'] 
        #                    if not item['label'].lower() == 'other']
        
        
        missed_labels = list(set(actual_labels) - set(prepared_labels)) 
        extra_labels = list(set(prepared_labels) - set(actual_labels))

        # if missed_labels != []:
        #     print(prepared_data)
        if not missed_labels == [] or not extra_labels == []:
            print("*"*10)
            print(labels)
            print(f"Missed labels: {missed_labels}")
            print(f"Extra labels: {extra_labels}")
            print("*"*10)
            print()

        # exit()

# for item in class_counts:
#      print(f"{item} ::: {class_counts[item]}")