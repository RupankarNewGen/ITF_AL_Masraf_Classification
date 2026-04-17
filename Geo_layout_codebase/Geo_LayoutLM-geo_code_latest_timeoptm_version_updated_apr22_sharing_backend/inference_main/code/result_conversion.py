import os
import json

# Function to transform the input dictionary
def transform_dict(input_dict):
    transformed_dict = {
        "GEOlayoutLMVForTokenClassification": {
            "keys_extraction": {},
            "keys_bboxes": {},
            "keys_confidence": {}
        }
    }
    
    # Iterate over the input dictionary, adding 'S-' in front of each key
    for key, value in input_dict.items():
        mapped_key = f"S-{key}"  # Add the prefix 'S-' to each key
        # Separate the values (text) and bounding boxes (bbox) in case of multiple entries
        extracted_values = [entry[0] for entry in value]  # List of all extracted text values
        bboxes = [entry[1] for entry in value]  # List of all bounding box coordinates
        confidence = [99.99] * len(bboxes) 
        # Update the transformed dictionary
        transformed_dict["GEOlayoutLMVForTokenClassification"]["keys_extraction"][mapped_key] = extracted_values
        transformed_dict["GEOlayoutLMVForTokenClassification"]["keys_bboxes"][mapped_key] = bboxes
        transformed_dict["GEOlayoutLMVForTokenClassification"]["keys_confidence"][mapped_key] = confidence
    
    return transformed_dict
import ast
# Function to process each .txt file in the folder
def process_files_in_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt') and 'lookup' not in filename and 'model_output' not in filename:  # Process only .txt files
            print('filename: %s' % filename)
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename.replace('.txt', '.json'))

            # Open and read the JSON-like content from the .txt file
            # with open(input_file_path, 'r') as file:
            #     data = json.load(file)
            with open(input_file_path, 'r') as file:
                data = file.read()  # Reads the entire file content
                # print(content)  # Display the content
            data = ast.literal_eval(data)
            # Transform the dictionary
            transformed_data = {}
            transformed_data[filename.replace('.txt', '.png')] = transform_dict(data)
            # Save the transformed data into the output folder
            with open(output_file_path, 'w') as output_file:
                json.dump(transformed_data, output_file, indent=4)

# Main function to specify input/output folders
def main():
    input_folder = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations/Eval_data/Results'
    output_folder = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations/Eval_data/final_results'
    os.makedirs(output_folder, exist_ok=True)
    process_files_in_folder(input_folder, output_folder)

if __name__ == '__main__':
    main()
