import os
import json
import csv

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

TARGET_SEARCH_FOLDERS = [
    "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/test_set2",
    "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/test_set_1",
    "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/DOC_3_splitted",
    "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/DOC_5_splitted"
]

SPLIT_RESULTS_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/New_test_data_jsons_splitted"

CLASS_MAPPING = {
    "Bill_of_Exchange": "BOE", "BOE": "BOE",
    "Bill_of_Lading": "BOL", "BOL": "BOL",
    "Commercial_Invoice": "CI", "CI": "CI",
    "Certificate_of_Origin": "COO", "COO": "COO",
    "Covering_Schedule": "CS", "CS": "CS",
    "Packing_List": "PL", "PL": "PL",
    "Others": "Others", "others": "Others"
}

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

IGNORE_GT_LIST = ["Others"]

def build_ground_truth_map():
    print("Scanning target folders to learn ground truth classifications...")
    image_to_class_map = {}
    
    for target_root in TARGET_SEARCH_FOLDERS:
        if not os.path.exists(target_root):
            print(f"  ⚠️ Target folder not found: {target_root}")
            continue
            
        for subdir_name in os.listdir(target_root):
            subdir_path = os.path.join(target_root, subdir_name)
            
            if os.path.isdir(subdir_path) and subdir_name in CLASS_MAPPING:
                standard_class_name = CLASS_MAPPING[subdir_name]
                
                for root, _, files in os.walk(subdir_path):
                    for img_file in files:
                        if os.path.splitext(img_file)[1].lower() in VALID_EXTS:
                            base_name = os.path.splitext(img_file)[0]
                            image_to_class_map[base_name] = standard_class_name

    print(f"✅ Learned ground truth for {len(image_to_class_map)} unique images.\n")
    return image_to_class_map

def main():
    gt_map = build_ground_truth_map()
    
    if not os.path.exists(SPLIT_RESULTS_DIR):
        print(f"❌ Split results directory not found: {SPLIT_RESULTS_DIR}")
        return
        
    workitem_dirs = [d for d in os.listdir(SPLIT_RESULTS_DIR) if os.path.isdir(os.path.join(SPLIT_RESULTS_DIR, d))]
    
    total_images = 0
    total_correct = 0
    
    final_aggregation_data = []
    
    mapping_diagnostics = {
        "Mapped_To_Class": {},
        "Mapped_To_Others_From_Folder": [],
        "Not_Found_Defaulted_To_Others": []
    }
    
    for workitem in workitem_dirs:
        workitem_path = os.path.join(SPLIT_RESULTS_DIR, workitem)
        jsons = [f for f in os.listdir(workitem_path) if f.endswith('.json')]
        
        if not jsons:
            continue
            
        csv_report_path = os.path.join(workitem_path, "evaluation_report.csv")
        
        folder_correct = 0
        folder_total = 0
        
        with open(csv_report_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Image Name", "Ground Truth Class", "Predicted Class", "Is Correct"])
            
            for json_filename in jsons:
                json_path = os.path.join(workitem_path, json_filename)
                
                # Derive base image name
                base_img_name = json_filename.replace('.json', '')
                
                # Fetch GT
                if base_img_name in gt_map:
                    gt_class = gt_map[base_img_name]
                    if gt_class.lower() == "others":
                        mapping_diagnostics["Mapped_To_Others_From_Folder"].append(base_img_name)
                    else:
                        if gt_class not in mapping_diagnostics["Mapped_To_Class"]:
                            mapping_diagnostics["Mapped_To_Class"][gt_class] = []
                        mapping_diagnostics["Mapped_To_Class"][gt_class].append(base_img_name)
                else:
                    gt_class = "Others"
                    mapping_diagnostics["Not_Found_Defaulted_To_Others"].append(base_img_name)
                
                # Fetch prediction
                pred_class = "UNKNOWN"
                with open(json_path, 'r', encoding='utf-8') as jf:
                    try:
                        data = json.load(jf)
                        # data structure: {"filename_page_1.jpeg": {"predicted_class": ...}}
                        if data:
                            first_key = list(data.keys())[0]
                            pred_class = data[first_key].get("predicted_class", "UNKNOWN")
                    except Exception as e:
                        print(f"Error reading JSON {json_path}: {e}")
                        
                is_correct = (gt_class.upper() == pred_class.upper())
                
                writer.writerow([base_img_name, gt_class, pred_class, "TRUE" if is_correct else "FALSE"])
                
                if gt_class not in IGNORE_GT_LIST:
                    if is_correct:
                        folder_correct += 1
                        total_correct += 1
                    folder_total += 1
                    total_images += 1
                
        folder_accuracy = (folder_correct / folder_total * 100) if folder_total > 0 else 0.0
        final_aggregation_data.append([workitem, folder_total, folder_correct, f"{folder_accuracy:.2f}%"])
        
        print(f"✓ Generated report for {workitem} ({folder_correct}/{folder_total} correct)")

    # Write the final aggregation CSV
    final_csv_path = os.path.join(SPLIT_RESULTS_DIR, "final_aggregation_report.csv")
    with open(final_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer_final = csv.writer(f)
        writer_final.writerow(["folder name", "total_page", "correct_page", "accuracy"])
        writer_final.writerows(final_aggregation_data)
        
    print(f"\n✅ Saved final aggregation report to: {final_csv_path}")

    print("\n==================================")
    print("✅ EVALUATION COMPLETION SUMMARY")
    print("==================================")
    print(f"Total Valid Images Evaluated: {total_images}")
    if total_images > 0:
         print(f"Overall Valid Accuracy: {(total_correct / total_images) * 100:.2f}%")
         
    diag_path = os.path.join(SPLIT_RESULTS_DIR, "mapping_diagnostics.json")
    with open(diag_path, 'w', encoding='utf-8') as df:
        json.dump(mapping_diagnostics, df, indent=4)
    print(f"✅ Saved detailed mapping diagnostics to: {diag_path}")
         
if __name__ == "__main__":
    main()
