import os
import json
import subprocess
import glob
import re

# ================= CONFIGURATION =================

# 1. Base Paths
BASE_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification"
MAIN_OUTPUT_FOLDER = os.path.join(BASE_DIR, "Data/Raw_ouput_data/images_for_inference") 
RESULTS_FOLDER = os.path.join(BASE_DIR, "Data/Raw_ouput_data/images_for_inference_results")
WORKING_DATA_DIR = os.path.join(BASE_DIR, "model_evaluation_codebase/extrction_benchmarking_codebase/Working_Data")

# 2. Scripts
INFERENCE_SCRIPT = os.path.join(BASE_DIR, "Inference/idp_full_pipeline_api_request_nested.py")
CONSTRUCTOR_SCRIPT = os.path.join(BASE_DIR, "Preprocessing/benchmarking_data_constructor.py")
YOLO_GT_SCRIPT = os.path.join(BASE_DIR, "model_evaluation_codebase/extrction_benchmarking_codebase/yolo_to_gt_converter.py")
REPORT_EVAL_SCRIPT = os.path.join(BASE_DIR, "model_evaluation_codebase/extrction_benchmarking_codebase/report_generator_rupankar_v3.py")

# 3. User configuration for classes mapping
# -> MANUALLY ADD YOUR PATHS BELOW <-
CLASSES_FILE_MAP = {
    "BOE": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations/BOE_annoation/classes.txt",
    "BOL": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations/BOL_annoation/classes_BOL.txt", 
    "CI": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations/CI_annoation/classes.txt",
    "COO": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations/COO_annoation/classes.txt",
    "CS": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations/CS_annoation/classes.txt",
    "PL": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations/PL_annoation/classesPL.txt"
}

# 4. Final Output JSON for Phase 2 Benchmarking Tracking
FINAL_RESULTS_JSON = os.path.join(WORKING_DATA_DIR, "benchmarking_final_results.json")

# =================================================

def run_script(cmd, cwd=None):
    env = os.environ.copy()
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=cwd, check=True)

def extract_accuracy_from_report(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'Accuracy:\s+([\d\.]+)%', content)
    if match:
        return float(match.group(1))
    return 0.0

def get_extraction_accuracy(report_dir):
    """Parses the report JSON (.txt) from report_generator to get OVERALL Complete_Match_Percentage."""
    # report_dir example: .../result_path/label_wise/BOE_2026-03-24_7/
    report_files = glob.glob(os.path.join(report_dir, "final_report_*.txt"))
    if not report_files:
        return None
        
    latest_report = max(report_files, key=os.path.getctime)
    try:
        with open(latest_report, 'r') as f:
            data = json.load(f)
        if "OVERALL" in data and "Complete_Match_Percentage" in data["OVERALL"]:
            return data["OVERALL"]["Complete_Match_Percentage"]
    except Exception as e:
        print(f"Error parsing {latest_report}: {e}")
    return None

def main():
    final_results = {}
    
    # Run Inference
    print("=== STEP 1: Running API Inference Pipeline ===")
    run_script(["/datadrive2/IDF_AL_MASRAF/qwen_env/bin/python", INFERENCE_SCRIPT])
    
    # Process Results
    print("\n=== STEP 2: Processing Inference Results ===")
    if not os.path.exists(RESULTS_FOLDER):
        print(f"Results folder not found: {RESULTS_FOLDER}")
        return
        
    for folder_name in os.listdir(RESULTS_FOLDER):
        folder_path = os.path.join(RESULTS_FOLDER, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        report_txt = os.path.join(folder_path, f"report_{folder_name}.txt")
        if not os.path.exists(report_txt):
            continue
            
        accuracy = extract_accuracy_from_report(report_txt)
        print(f"Document {folder_name} -> Inference Accuracy: {accuracy}%")
        
        if accuracy >= 80.0:
            print(f"  [+] Qualified for extraction benchmarking.")
            final_results[folder_name] = {
                "Overall_Inference_Accuracy": accuracy,
                "Classes": {}
            }
            
            # Run constructor
            output_folder = os.path.join(WORKING_DATA_DIR, folder_name)
            run_script([
                "/datadrive2/IDF_AL_MASRAF/qwen_env/bin/python", CONSTRUCTOR_SCRIPT,
                "--json_folder", folder_path,
                "--report_path", report_txt,
                "--output_folder", output_folder
            ])
            
            # Process each class
            if os.path.exists(output_folder):
                for class_name in os.listdir(output_folder):
                    class_dir = os.path.join(output_folder, class_name)
                    if not os.path.isdir(class_dir) or class_name not in CLASSES_FILE_MAP:
                        continue
                        
                    print(f"  -> Processing class: {class_name}")
                    labels_dir = os.path.join(class_dir, "labels")
                    img_dir = os.path.join(MAIN_OUTPUT_FOLDER, folder_name, class_name)
                    master_data_dir = os.path.join(class_dir, "New_Master_Data_Merged")
                    
                    # Convert YOLO to GT
                    run_script([
                        "/datadrive2/IDF_AL_MASRAF/qwen_env/bin/python", YOLO_GT_SCRIPT,
                        "--yolo_dir", labels_dir,
                        "--img_dir", img_dir,
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
                    
                    # Parse evaluation score
                    label_wise_dir = os.path.join(class_dir, "result_path", "label_wise")
                    if os.path.exists(label_wise_dir):
                        subdirs = [os.path.join(label_wise_dir, d) for d in os.listdir(label_wise_dir) if os.path.isdir(os.path.join(label_wise_dir, d))]
                        if subdirs:
                            latest_eval_dir = max(subdirs, key=os.path.getctime)
                            ext_acc = get_extraction_accuracy(latest_eval_dir)
                            if ext_acc is not None:
                                final_results[folder_name]["Classes"][class_name] = ext_acc
                                print(f"  -> {class_name} Extraction Accuracy: {ext_acc}%")
                            else:
                                print(f"  -> {class_name} Could not parse extraction accuracy.")
        else:
            print(f"  [-] Disqualified (<80%).")

    # Save tracking JSON
    os.makedirs(WORKING_DATA_DIR, exist_ok=True)
    with open(FINAL_RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\n✅ Phase 2 completed successfully. Master results saved to {FINAL_RESULTS_JSON}")

if __name__ == "__main__":
    main()
