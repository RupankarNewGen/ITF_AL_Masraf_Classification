import os
import json
import pandas as pd
from ast import literal_eval
from fuzzywuzzy import fuzz

def process_files(folder_path):
    # List to store all rows for the DataFrame
    all_data = []
    
    # Iterate through all txt files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                content = file.read()
                # Convert string representation of dictionary to actual dictionary
                try:
                    data = literal_eval(content)
                    
                    # Process each key and its values
                    for key, values in data.items():
                        # If values is a list of lists (multiple entries for the key)
                        if isinstance(values, list):
                            for value in values:
                                row = {
                                    'filename': filename,
                                    'key': key,
                                    'value': value[0],
                                    'bbox': str(value[1]),  # Convert bbox list to string
                                    'confidence': value[2],
                                    'updated_value': value[3]
                                }
                                all_data.append(row)
                                
                except (SyntaxError, ValueError) as e:
                    print(f"Error processing file {filename}: {e}")
    
    # Create DataFrame from all collected data
    df = pd.DataFrame(all_data)
    
    # Apply fuzzy matching and create new column
    def fuzzy_compare(row):
        # Convert both values to lowercase strings for better matching
        val1 = str(row['value']).lower()
        val2 = str(row['updated_value']).lower()
        val1 = val1.replace(' ', '')
        val2 = val2.replace(' ', '')
        # Get fuzzy match ratio
        ratio = fuzz.ratio(val1, val2)
        
        # If ratio is very high (exact or near match), return the value
        # Otherwise return 0
        if ratio == 100:  # You can adjust this threshold
            # return row['value']
            return 1
        else:
            return 0
    
    # Add new column with fuzzy matching results
    df['matched_result'] = df.apply(fuzzy_compare, axis=1)
    
    # Save to Excel
    output_file = 'processed_data_with_matching.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Data has been saved to {output_file}")
    
    return df

# Example usage
if __name__ == "__main__":
    # Replace with your folder path
    folder_path = "/home/ntlpt19/Desktop/TF_release/geolm_api/debug_ocr/temp_res"
    df = process_files(folder_path)
    
    # Display first few rows of the DataFrame
    print("\nFirst few rows of the processed data:")
    print(df.head())
    
    # Display some statistics about matches
    total_rows = len(df)
    matches = len(df[df['matched_result'] != '0'])
    print(f"\nMatching Statistics:")
    print(f"Total rows: {total_rows}")
    print(f"Matched rows: {matches}")
    print(f"Non-matched rows: {total_rows - matches}")