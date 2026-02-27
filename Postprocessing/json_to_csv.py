import os
import json
import csv
from tqdm import tqdm
from collections import Counter # Added for class counting

def generate_classification_csv(json_folder, output_csv_path):
    """
    Reads JSON files from the specified folder and creates a CSV
    with 'image name', 'class', and 'reasoning' columns.
    """
    data_to_save = []
    class_counter = Counter() # Initialize the counter
    
    # Check if folder exists
    if not os.path.exists(json_folder):
        print(f"Error: The folder {json_folder} does not exist.")
        return

    # List all JSON files
    files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files. Starting CSV generation...")

    for filename in tqdm(files, desc="Parsing JSONs"):
        file_path = os.path.join(json_folder, filename)
        
        # Get image name from the JSON filename (removing .json extension)
        image_name = os.path.splitext(filename)[0]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
                # Extract values with safety defaults
                classification = content.get("classification", "N/A")
                reasoning = content.get("reasoning", "No reasoning provided")
                
                # Update the count for this specific class
                class_counter[classification] += 1
                
                data_to_save.append({
                    "image name": image_name,
                    "class": classification,
                    "reasoning": reasoning
                })
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    # Write to CSV
    headers = ["image name", "class", "reasoning"]
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_to_save)
        
        print("\n" + "-"*30)
        print(f"SUCCESS: CSV saved at {output_csv_path}")
        print(f"Total Rows: {len(data_to_save)}")
        print("-" * 30)
        
        # --- NEW: Print data count for each class ---
        print("\nCLASS DISTRIBUTION:")
        # Sorting by count (highest first) makes it easier to read
        for label, count in class_counter.most_common():
            print(f"{label:<25} ----> {count}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    JSON_INPUT_FOLDER = "/datadrive2/IDF_AL_MASRAF/532_first_set_thinking_model_result"  # Folder containing the JSON files
    FINAL_CSV_PATH = "/datadrive2/IDF_AL_MASRAF/532_first_set_thinking_model_result.csv"

    generate_classification_csv(JSON_INPUT_FOLDER, FINAL_CSV_PATH)