import json
import os
sample = {
    "/datadrive2/ritwick/test/2_page_0.png": {
        "layoutLMV2ForTokenClassification": {
            "keys_extraction": {
                "invoice_number": [
                    "ve / tn / 21-22 / 019", "ve / tn / 21-22 / 019"
                ],
                "vendor_name": [
                    "vedha electricals"
                ],
                "invoice_date": [
                    "26 - jul - 21"
                ],
                "invoice_total": [
                    "25,134.00"
                ]
            },
            "keys_bboxes": {
                "invoice_number": [
                    [
                        'rakesh',
                        133.19,
                        706.8,
                        161.23
                    ],
                    [
                        406.72,
                        133.19,
                        706.8,
                        161.23
                    ]
                ],
                "vendor_name": [
                    [
                        890.32,
                        238.34,
                        1306.96,
                        269.885
                    ]
                ],
                "invoice_date": [
                    [
                        1899.68,
                        133.19,
                        2063.36,
                        161.23
                    ]
                ],
                "invoice_total": [
                    [
                        1847.6,
                        1934.76,
                        2053.44,
                        1976.82
                    ]
                ]
            },
            "keys_confidence": {
                "invoice_number": [
                    97.48736737053345,   97.48736737053345
                ],
                "vendor_name": [
                    98.08666666666667
                ],
                "invoice_date": [
                    98.09355939214505
                ],
                "invoice_total": [
                    94.48
                ]
            }
        }
    }
}


# Function to reverse the transformed dictionary (result2) back to original format (result1)
def reverse_transform_dict(input_dict):
    reversed_dict = {}
    
    # Access the required nested dictionaries from the input structure
    keys_extraction = input_dict["layoutLMV2ForTokenClassification"]["keys_extraction"]
    keys_bboxes = input_dict["layoutLMV2ForTokenClassification"]["keys_bboxes"]

    # Iterate through the keys in keys_extraction
    for key in keys_extraction:
        # Remove the prefix 'S-' from the key (if applicable)
        original_key = key
        
        # Extract the corresponding text and bounding boxes
        extracted_values = keys_extraction[key]  # List of extracted text values
        bboxes = keys_bboxes[key]  # List of bounding boxes
        
        # Combine the extracted values and bounding boxes to match the reversed format
        # Initialize the list for the current key in the reversed dictionary
        reversed_dict[original_key] = []
        
        # Iterate over the extracted values and their corresponding bounding boxes
        for value, bbox in zip(extracted_values, bboxes):
            reversed_dict[original_key].append([value, bbox])
    
    return reversed_dict


# Function to process each .json file in the folder
def process_json_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):  # Process only .json files
            print('Processing file: %s' % filename)
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename.replace('.json', '.txt'))

            # Open and read the JSON content from the file
            with open(input_file_path, 'r') as file:
                data = json.load(file)
            # data = sample
            # Extract the main nested dictionary (the first key is the filename)
            page_key = list(data.keys())[0]
            transformed_data = data[page_key]

            # Reverse the transformation
            reversed_data = reverse_transform_dict(transformed_data)
            # print(reversed_data)
            # exit('ok')
            # Save the reversed data into the output folder as .txt (or as needed)
            with open(output_file_path, 'w') as output_file:
                json.dump(reversed_data, output_file)#, indent=1)
            print(f"Reversed result saved to {output_file_path}")

# Main function to specify input/output folders
def main():
    input_folder = '/home/ntlpt19/Desktop/TF_release/geolm_api/Testing_ingram_samples/Inv_Lmv2_Json_Results'  # Replace with your input folder path
    output_folder = '/home/ntlpt19/Desktop/TF_release/geolm_api/Testing_ingram_samples/Results_Images_LMV2'  # Replace with your output folder path
    process_json_files(input_folder, output_folder)

if __name__ == '__main__':
    main()
