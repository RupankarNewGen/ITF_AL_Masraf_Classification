import os
import json
import pandas as pd
from ast import literal_eval
from fuzzywuzzy import fuzz



            
def find_ocr_text_in_bbox(target_bbox, ocr_data):
    """
    Find all OCR text entries that fall within a target bounding box.
    
    Args:
        target_bbox (list): [x1, y1, x2, y2] coordinates of target box
        ocr_data (dict): Dictionary of OCR data with bounding boxes
    
    Returns:
        str: Concatenated text of all matching OCR entries
    """
    matching_text = []
    
    def is_within_bbox(box1, box2):
        # Check if box1 is within or overlaps significantly with box2
        x1, y1, x2, y2 = box1
        tx1, ty1, tx2, ty2 = box2
        
        # Calculate overlap area
        x_left = max(x1, tx1)
        y_top = max(y1, ty1)
        x_right = min(x2, tx2)
        y_bottom = min(y2, ty2)
        
        if x_right > x_left and y_bottom > y_top:
            overlap_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (x2 - x1) * (y2 - y1)
            overlap_ratio = overlap_area / box1_area
            return overlap_ratio > 0.5
        return False

    # Find all OCR entries that fall within the target bbox
    for ocr_id, ocr_entry in ocr_data.items():
        ocr_bbox = ocr_entry['bbox']
        if is_within_bbox(ocr_bbox, target_bbox):
            matching_text.append(ocr_entry['text'])
    
    return ' '.join(matching_text)

def process_results_with_ocr(results, ocr_data):
    """
    Process results dictionary and add OCR text for each bbox entry.
    
    Args:
        results (dict): Dictionary containing results with bboxes
        ocr_data (dict): Dictionary of OCR data
    
    Returns:
        dict: Updated results dictionary with OCR text
    """
    processed_results = {}
    
    for key, value_list in results.items():
        processed_results[key] = []
        for entry in value_list:
            # Each entry is [text, bbox, confidence]
            text, bbox, confidence = entry
            ocr_text = find_ocr_text_in_bbox(bbox, ocr_data)
            # Get fuzzy match ratio
            ratio = fuzz.ratio(str(text).lower().replace(' ', ''), str(ocr_text).lower().replace(' ', ''))
            # If ratio is very high (exact or near match), return the value
            # Otherwise return 0
            if ratio == 100:  # You can adjust this threshold
                processed_results[key].append([text, bbox, confidence, ocr_text])
            else:
                processed_results[key].append([text, bbox, confidence, text])
                
    return processed_results

from pre_process_utility import gv_data

def process_files(folder_path):
    # List to store all rows for the DataFrame
    all_data = []
    
    # Iterate through all txt files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('model_output.txt'):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                content = file.read()
            file.close()
                # Convert string representation of dictionary to actual dictionary
            data = literal_eval(content)
            up_filename = filename.replace('1model_output.txt', '_textAndCoordinates.txt')
            all_words_ = gv_data(file_path, ocr_file = os.path.join(ocr_folder,up_filename))
            updated_result = process_results_with_ocr(data, all_words_)
            with open(os.path.join(result_path, up_filename), 'w') as frs_:
                frs_.write(str(updated_result))
            frs_.close()


# Example usage
if __name__ == "__main__":
    # Replace with your folder path
    folder_path = "/home/ntlpt19/Desktop/TF_release/geolm_api/debug_ocr/temp_testing"
    ocr_folder = "/home/ntlpt19/Desktop/TF_release/geolm_api/Ingram_POC_Samples/OCR"
    result_path = '/home/ntlpt19/Desktop/TF_release/geolm_api/debug_ocr/temp_res'
    df = process_files(folder_path)
    
