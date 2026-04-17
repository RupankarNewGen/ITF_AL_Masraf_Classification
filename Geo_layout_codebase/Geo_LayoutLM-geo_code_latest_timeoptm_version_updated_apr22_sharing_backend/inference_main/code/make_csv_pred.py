import os
import json
import csv

def process_json_folder(input_folder, output_csv):
    # Define the header for the CSV file
    csv_header = ["image_name", "label_name", "value", "bbox", "confidence"]

    # Create an empty list to hold the rows for the CSV
    csv_rows = []

    # Iterate through all the files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            
            # Open and load the JSON file
            with open(file_path, "r") as file:
                data = json.load(file)

            # Extract the image name
            image_name = filename.replace(".json", "") + ".png"

            # Navigate through the JSON structure to retrieve relevant data
            for key, values in data.get(image_name, {}).get("GEOlayoutLMVForTokenClassification", {}).items():
                if key == "keys_extraction":
                    keys_extraction = values
                elif key == "keys_bboxes":
                    keys_bboxes = values
                elif key == "keys_confidence":
                    keys_confidence = values
            
            # Process each label in keys_extraction
            for label_name, label_values in keys_extraction.items():
                bboxes = keys_bboxes.get(label_name, [])
                confidences = keys_confidence.get(label_name, [])
                
                # Ensure alignment of values, bboxes, and confidences
                for i, value in enumerate(label_values):
                    bbox = bboxes[i] if i < len(bboxes) else None
                    confidence = confidences[i] if i < len(confidences) else None
                    
                    # Add a row to the CSV rows list
                    csv_rows.append([
                        image_name, 
                        label_name.removeprefix('S-'),
                        value, 
                        bbox, 
                        confidence
                    ])

    # Write rows to the CSV file
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_header)  # Write the header
        csv_writer.writerows(csv_rows)  # Write the data rows

# Example usage
input_folder = "/home/ntlpt19/Desktop/TF_release/geolm_api/nakilat_poc/itr2/final_results"
output_csv = "pred_output_dec17.csv"
process_json_folder(input_folder, output_csv)
