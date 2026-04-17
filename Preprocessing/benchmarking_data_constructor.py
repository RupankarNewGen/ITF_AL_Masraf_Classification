import os
import shutil

def organize_results_and_labels(json_folder, report_path, label_search_folders, output_folder):
    # ---------------------------------------------------------
    # STEP 1: Parse the report to find correctly predicted images
    # ---------------------------------------------------------
    print(f"Reading report: {report_path}...")
    
    if not os.path.exists(report_path):
        print(f"❌ Error: Report file not found at {report_path}")
        return

    correct_predictions = []
    
    with open(report_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # We only care about lines indicating a correct prediction
            if line.startswith('[✓]'):
                try:
                    # Example line: [✓] image_name.jpeg: Pred=CS | GT=CS
                    # Split at the first colon to separate filename from the stats
                    parts = line.split(':', 1)
                    
                    # Extract and clean the image name
                    img_filename = parts[0].replace('[✓]', '').strip()
                    base_name = os.path.splitext(img_filename)[0] # Removes .jpeg/.png
                    
                    # Extract the predicted class
                    stats_part = parts[1] # " Pred=CS | GT=CS"
                    pred_string = stats_part.split('|')[0].strip() # "Pred=CS"
                    predicted_class = pred_string.replace('Pred=', '').strip() # "CS"
                    
                    correct_predictions.append({
                        'base_name': base_name,
                        'class': predicted_class
                    })
                except Exception as e:
                    print(f"  ⚠️ Could not parse line: {line} | Error: {e}")

    print(f"✅ Found {len(correct_predictions)} correct predictions to process.\n")

    # ---------------------------------------------------------
    # STEP 2: Build a master index of all available label files
    # ---------------------------------------------------------
    print("Scanning label folders to build an index...")
    label_file_map = {}
    
    for folder in label_search_folders:
        if not os.path.exists(folder):
            print(f"  ⚠️ Label folder not found: {folder}")
            continue
            
        # os.walk searches the folder and all subfolders recursively
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.txt'):
                    label_file_map[file] = os.path.join(root, file)
                    
    print(f"✅ Indexed {len(label_file_map)} label files.\n")

    # ---------------------------------------------------------
    # STEP 3: Organize JSONs and Labels into the output directory
    # ---------------------------------------------------------
    print("Organizing files into the output structure...")
    success_count = 0
    missing_json_count = 0
    missing_label_count = 0

    for item in correct_predictions:
        base_name = item['base_name']
        predicted_class = item['class']
        
        # Define the source file names
        json_filename = f"{base_name}.json"
        txt_filename = f"{base_name}.txt"
        
        # Define the source paths
        src_json_path = os.path.join(json_folder, json_filename)
        src_txt_path = label_file_map.get(txt_filename) # Will be None if not found
        
        # Define the destination directories
        dest_class_dir = os.path.join(output_folder, predicted_class)
        dest_results_dir = os.path.join(dest_class_dir, 'Results_Images')
        dest_labels_dir = os.path.join(dest_class_dir, 'labels')
        
        # Create directories safely
        os.makedirs(dest_results_dir, exist_ok=True)
        os.makedirs(dest_labels_dir, exist_ok=True)
        
        # Copy the JSON file
        if os.path.exists(src_json_path):
            shutil.copy2(src_json_path, os.path.join(dest_results_dir, json_filename))
        else:
            print(f"  ⚠️ Missing JSON for: {base_name}")
            missing_json_count += 1
            
        # Copy the Label file
        if src_txt_path and os.path.exists(src_txt_path):
            shutil.copy2(src_txt_path, os.path.join(dest_labels_dir, txt_filename))
        else:
            print(f"  ⚠️ Missing Label (.txt) for: {base_name}")
            missing_label_count += 1
            
        success_count += 1

    # ---------------------------------------------------------
    # STEP 4: Summary
    # ---------------------------------------------------------
    print("-" * 50)
    print("=== ORGANIZATION COMPLETE ===")
    print(f"Processed items: {success_count}")
    print(f"Missing JSONs:   {missing_json_count}")
    print(f"Missing Labels:  {missing_label_count}")
    print(f"Output Location: {output_folder}")
    print("-" * 50)


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Organize results and labels into class folders.")
    parser.add_argument("--json_folder", required=True, help="Folder containing generated JSON files")
    parser.add_argument("--report_path", required=True, help="Path to the report.txt file")
    parser.add_argument("--output_folder", required=True, help="Where to build final organized structure")
    parser.add_argument("--config_path", required=True, help="Path to configuration JSON file")
    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        config = json.load(f)
        
    # 3. A list of folders where the ground-truth label .txt files might live
    # (The script will search these recursively)
    LABEL_SEARCH_FOLDERS = config.get("directories", {}).get("LABEL_SEARCH_FOLDERS", [])
    
    organize_results_and_labels(
        json_folder=args.json_folder,
        report_path=args.report_path,
        label_search_folders=LABEL_SEARCH_FOLDERS,
        output_folder=args.output_folder
    )