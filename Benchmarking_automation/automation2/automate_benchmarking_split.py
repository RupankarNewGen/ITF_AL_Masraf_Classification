import os
import json
import csv
import subprocess
import shutil
import glob
import pandas as pd

# ================= CONFIGURATION =================

# Load config.json mapping
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

BASE_DIR = config["directories"]["BASE_DIR"]
SPLIT_RESULTS_DIR = config["directories"]["SPLIT_RESULTS_DIR"]
WORKING_DATA_DIR = config["directories"]["WORKING_DATA_DIR"]
OCR_DIR = config["directories"]["OCR_DIR"]

CONSTRUCTOR_SCRIPT = os.path.join(BASE_DIR, "Preprocessing/benchmarking_data_constructor.py")
YOLO_GT_SCRIPT = os.path.join(BASE_DIR, "model_evaluation_codebase/extrction_benchmarking_codebase/yolo_to_gt_converter.py")
REPORT_EVAL_SCRIPT = os.path.join(BASE_DIR, "model_evaluation_codebase/extrction_benchmarking_codebase/report_generator_rupankar_v3.py")

CLASSES_FILE_MAP = config["classes_file_map"]
TARGET_SEARCH_FOLDERS = config["directories"]["TARGET_SEARCH_FOLDERS"]

def run_script(cmd, cwd=None):
    env = os.environ.copy()
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=cwd, check=True)

def build_image_index():
    print("Building global image path index...")
    img_index = {}
    valid_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    for target_root in TARGET_SEARCH_FOLDERS:
        if not os.path.exists(target_root): 
            continue
        # Use os.walk to recursively search all subdirectories
        for root, dirs, files in os.walk(target_root):
            for img_file in files:
                if os.path.splitext(img_file)[1].lower() in valid_exts:
                    base_name = os.path.splitext(img_file)[0]
                    img_index[base_name] = os.path.join(root, img_file)
    print(f"✅ Indexed {len(img_index)} physical images.")
    return img_index

def main():
    img_index = build_image_index()
    os.makedirs(WORKING_DATA_DIR, exist_ok=True)
    
    workitem_dirs = [d for d in os.listdir(SPLIT_RESULTS_DIR) if os.path.isdir(os.path.join(SPLIT_RESULTS_DIR, d))]
    
    total_valid = 0
    
    for workitem in workitem_dirs:
        workitem_path = os.path.join(SPLIT_RESULTS_DIR, workitem)
        csv_report_path = os.path.join(workitem_path, "evaluation_report.csv")
        
        if not os.path.exists(csv_report_path):
            continue
            
        valid_images = []
        
        # Read the local CSV
        with open(csv_report_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["Image Name"]
                gt_class = row["Ground Truth Class"]
                is_correct = row["Is Correct"]
                
                # Filter strictly for valid
                if gt_class != "Others" and is_correct == "TRUE":
                    valid_images.append((img_name, gt_class))
                    
        # Skip if no valid images in this workitem
        if not valid_images:
            continue
            
        print(f"\n=== Processing Workitem: {workitem} ({len(valid_images)} Valid Images) ===")
        total_valid += len(valid_images)
        
        # 1. Create Dummy Report specific to evaluating strictly the valid images
        dummy_report_path = os.path.join(workitem_path, "dummy_report.txt")
        with open(dummy_report_path, 'w', encoding='utf-8') as f:
            for img_name, gt_class in valid_images:
                f.write(f"[✓] {img_name}.jpeg: Pred={gt_class} | GT={gt_class}\n")
                
        # 2. Run benchmarking constructor
        output_folder = os.path.join(WORKING_DATA_DIR, workitem)
        run_script([
            "/datadrive2/IDF_AL_MASRAF/qwen_env/bin/python", CONSTRUCTOR_SCRIPT,
            "--json_folder", workitem_path,
            "--report_path", dummy_report_path,
            "--output_folder", output_folder,
            "--config_path", config_path
        ])
        
        # 3. Supply standard images and run downstream extractors
        if os.path.exists(output_folder):
            for class_name in os.listdir(output_folder):
                class_dir = os.path.join(output_folder, class_name)
                if not os.path.isdir(class_dir) or class_name not in CLASSES_FILE_MAP:
                    continue
                    
                print(f"  -> Extractor Pipeline for: {class_name}")
                
                # Create a local images directory to satisfy converter
                local_img_dir = os.path.join(class_dir, "images")
                os.makedirs(local_img_dir, exist_ok=True)
                
                # Link required images using our index
                labels_dir = os.path.join(class_dir, "labels")
                if os.path.exists(labels_dir):
                    for label_file in os.listdir(labels_dir):
                        base_img_name = label_file.replace('.txt', '')
                        if base_img_name in img_index:
                            src_img_path = img_index[base_img_name]
                            dest_img_path = os.path.join(local_img_dir, os.path.basename(src_img_path))
                            shutil.copy2(src_img_path, dest_img_path)
                
                master_data_dir = os.path.join(class_dir, "New_Master_Data_Merged")
                
                # Convert YOLO to GT
                run_script([
                    "/datadrive2/IDF_AL_MASRAF/qwen_env/bin/python", YOLO_GT_SCRIPT,
                    "--yolo_dir", labels_dir,
                    "--ocr_dir", OCR_DIR,
                    "--img_dir", local_img_dir,
                    "--classes_file", CLASSES_FILE_MAP[class_name],
                    "--output_dir", master_data_dir
                ])
                
                # Run Evaluator script
                eval_cwd = os.path.join(BASE_DIR, "model_evaluation_codebase/extrction_benchmarking_codebase")
                run_script([
                    "/datadrive2/IDF_AL_MASRAF/qwen_env/bin/python", REPORT_EVAL_SCRIPT,
                    "--doc_code", class_name,
                    "--folder_path", class_dir
                ], cwd=eval_cwd)
                
                # Optionally delete copied images and labels according to config
                if not config.get("settings", {}).get("save_images_and_labels", True):
                    print(f"  -> Cleaning up temporarily copied images and labels for {class_name}")
                    if os.path.exists(local_img_dir):
                        shutil.rmtree(local_img_dir)
                    if os.path.exists(labels_dir):
                        shutil.rmtree(labels_dir)
        
        # --- AGGREGATE NATIVE CSVS FOR WORKITEM DOCUMENT ---
        def build_page_wise_summary(csv_pattern_str, output_suffix):
            native_csvs = glob.glob(csv_pattern_str)
            if native_csvs:
                df_list = []
                for csv_file in native_csvs:
                    try:
                        df = pd.read_csv(csv_file)
                        base_csv_name = os.path.basename(csv_file)
                        class_name_extracted = base_csv_name.split('_')[0]
                        df['Class_Name'] = class_name_extracted
                        df_list.append(df)
                    except Exception as e:
                        print(f"Failed to read {csv_file}: {e}")
                
                if df_list:
                    master_df = pd.concat(df_list, ignore_index=True)
                    page_data = []
                    for file_name, file_df in master_df.groupby("File_Name"):
                        # Drop fields where the Ground Truth actual text is completely empty or NaN
                        valid_df = file_df.dropna(subset=['actual']).copy()
                        valid_df = valid_df[valid_df['actual'].astype(str).str.strip() != ""]
                        valid_df = valid_df[valid_df['actual'].astype(str).str.strip().str.lower() != "nan"]
                        
                        page_class = file_df['Class_Name'].iloc[0] if 'Class_Name' in file_df.columns and not file_df.empty else "UNKNOWN"

                        # Filter evaluation fields if enabled and configured
                        if config.get("settings", {}).get("enable_field_filtering", False):
                            allowed_fields = config.get("evaluation_fields", {}).get(page_class, [])
                            if allowed_fields and "label_name" in valid_df.columns:
                                valid_df = valid_df[valid_df['label_name'].isin(allowed_fields)]
                        
                        total_fields = len(valid_df)
                        valid_df["Match/No_Match"] = pd.to_numeric(valid_df["Match/No_Match"], errors='coerce').fillna(0)
                        matched_fields = valid_df["Match/No_Match"].sum()
                        page_acc = (matched_fields / total_fields * 100) if total_fields > 0 else 0
                        page_class = file_df['Class_Name'].iloc[0] if 'Class_Name' in file_df.columns and not file_df.empty else "UNKNOWN"
                        page_data.append([file_name, page_class, total_fields, matched_fields, page_acc])
                    
                    doc_total = sum(item[2] for item in page_data)
                    doc_matched = sum(item[3] for item in page_data)
                    doc_acc = (doc_matched / doc_total * 100) if doc_total > 0 else 0
                    page_data.append(["OVERALL_DOCUMENT", "ALL", doc_total, doc_matched, doc_acc])
                    
                    summary_df = pd.DataFrame(page_data, columns=["File_Name", "Class_Name", "Total_Fields_Present", "Total_Fields_Matched", "Accuracy_Percentage"])
                    final_summary_path = os.path.join(output_folder, f"{workitem}{output_suffix}")
                    summary_df.to_csv(final_summary_path, index=False)
                    print(f"  -> Generated Master Document Report: {final_summary_path}")

        # 1. Native Pipeline File Aggregation
        csv_pattern_native = os.path.join(output_folder, "*", "result_path", "*", "*_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv")
        build_page_wise_summary(csv_pattern_native, "_final_document_page_wise_summary.csv")
        
        # 2. Dynamic Fuzzy Uniform Aggregation based on config threshold
        thresh = config.get("settings", {}).get("fuzzy_threshold", 75)
        csv_pattern_thresh = os.path.join(output_folder, "*", "result_path", "*", f"*_analysis_fuzzy_{thresh}.csv")
        build_page_wise_summary(csv_pattern_thresh, f"_final_document_page_wise_summary_{thresh}.csv")

    print(f"\n✅ Benchmarking Orchestrator Completed Successfully! Total highly confident single-pages processed: {total_valid}")

if __name__ == "__main__":
    main()
