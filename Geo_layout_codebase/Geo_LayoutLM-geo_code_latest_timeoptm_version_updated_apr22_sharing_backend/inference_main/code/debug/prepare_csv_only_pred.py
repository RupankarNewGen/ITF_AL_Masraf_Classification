import os
import json
import ast
import pandas as pd
from typing import Dict, List, Tuple

def transform_dict(input_dict: Dict, filename: str) -> List[Dict]:
    """
    Transform input dictionary into a list of dictionaries suitable for DataFrame
    Each dictionary represents a row in the final Excel
    """
    print("Processing dictionary for:", filename)
    print(input_dict)
    print('>>>>>>>>>>>>>>>>>>')
    
    transformed_rows = []
    
    # Iterate over the input dictionary
    for key, value in input_dict.items():
        # mapped_key = f"S-{key}"  # Add the prefix 'S-' to each key
        mapped_key = key  # Add the prefix 'S-' to each key
        
        # Extract values, bboxes, and generate confidence scores
        for entry in value:
            # Explicitly keep as string to preserve leading zeros
            extracted_value = str(entry[0])
            bbox = entry[1]
            conf = entry[2] if len(entry) > 2 else 99.99
            
            row = {
                'file_name': filename,
                'key': mapped_key,
                'key_value': extracted_value,
                'key_bboxes': str(bbox),
                'key_confidence': conf
            }
            transformed_rows.append(row)
    
    return transformed_rows

def process_files_in_folder(input_folder: str, output_excel_path: str):
    """
    Process all txt files in the input folder and save results to a single Excel file
    """
    all_rows = []
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt') and 'lookup' not in filename and 'model_output' not in filename:
            print(f'Processing filename: {filename}')
            input_file_path = os.path.join(input_folder, filename)
            print('$$$$$$$$$$$$$#####$$$$$$$$', input_file_path)
            # Read and parse the file content
            with open(input_file_path, 'r') as file:
                data = ast.literal_eval(file.read())
            
            # Get base filename without extension for image reference
            image_filename = filename.replace('.txt', '.png')
            
            # Transform the data and collect rows
            transformed_rows = transform_dict(data, image_filename)
            all_rows.extend(transformed_rows)
    
    # Create DataFrame from all collected rows
    df = pd.DataFrame(all_rows)
    
    # Ensure the key_value column is treated as string to preserve leading zeros
    df['key_value'] = df['key_value'].astype(str)
    
    # Write to Excel with string formatting
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Extraction Results')
        
        # Get the workbook and the worksheet
        workbook = writer.book
        worksheet = writer.sheets['Extraction Results']
        
        # Set the column width
        for idx, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).apply(len).max(), len(col))
            worksheet.set_column(idx, idx, max_length + 2)

def main():
    input_folder = '/home/ntlpt19/Desktop/TF_release/geolm_api/Lesotho_NIC/annotation_validated/NIC_data/Eval_data/Pred_Results'
    output_excel = '/home/ntlpt19/Desktop/TF_release/geolm_api/Lesotho_NIC/annotation_validated/NIC_data/Eval_data/final_results.xlsx'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    
    # Process files and create Excel
    process_files_in_folder(input_folder, output_excel)
    print(f"Excel file created successfully at: {output_excel}")

if __name__ == '__main__':
    main()