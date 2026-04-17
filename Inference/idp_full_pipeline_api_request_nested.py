import os
import json
import time
import requests

# ================= CONFIGURATION =================

#API_URL = "http://10.2.0.35:8021/bundleAPI/ntdemoinstance_outlook_com_1773831699-56428b2d-f969-4a02-97f9-2e5766bdf1b1/V3/1774334159680"
API_URL = "http://10.2.0.36:8021/bundleAPI/ntdemoinstance_outlook_com_1773831699-56428b2d-f969-4a02-97f9-2e5766bdf1b1/V3/1774334159680"
#STATUS_API_BASE_URL = "http://10.2.0.35:8021/task-status/"
STATUS_API_BASE_URL = "http://10.2.0.36:8021/task-status/"
# The main folder containing the TF... case folders
INPUT_PATH = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/images_for_inference"

# Where to save the output JSONs and the reports
OUTPUT_FOLDER = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/images_for_inference_results"

HEADERS = {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhdXRoZW50aWNhdGVkX3VzZXIiLCJleHAiOjE3OTU5MTQyMTR9.vka3MnQGXe2qeOAOico0AtQsXEqZNUqVAzaJPbt5a-s'
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

POLL_INTERVAL_SECONDS = 5
MAX_RETRIES = 60 

# =================================================

def poll_task_status(task_id):
    """Loops and checks the status API until the task is complete. Returns the JSON result."""
    status_url = f"{STATUS_API_BASE_URL}{task_id}"
    print(f"   ↳ Task ID received: {task_id}")
    print("   ↳ Polling for results", end="", flush=True)

    for _ in range(MAX_RETRIES):
        try:
            response = requests.get(status_url, headers=HEADERS, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "").upper()
                
                if status in ["PENDING", "PROCESSING", "STARTED"]:
                    print(".", end="", flush=True)
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                
                elif status in ["SUCCESS", "COMPLETED", "DONE"] or data.get("result") is not None:
                    print(" ✓ Done!")
                    return data.get("result", data)
                
                elif status in ["FAILED", "FAILURE", "ERROR"]:
                    print(f" ❌ Server failed to process the image. Status: {status}\n")
                    return None
                else:
                    print(f" ⚠️ Unknown status received: {status}\n")
                    return None
            else:
                print(f" ❌ Error checking status. HTTP {response.status_code}\n")
                return None
                
        except Exception as e:
            print(f" ❌ Polling request failed: {str(e)}\n")
            return None

    print(" ❌ Timeout reached!\n")
    return None

def process_single_image(img_path):
    """Submits a single image and returns the final JSON payload."""
    filename = os.path.basename(img_path)
    print(f"Processing: {filename}")

    payload_dict = {
        "fileType": "img",
        "image_name": filename,
        "ocr_flag": "false",
        "other_params": {"product_name": "Import Bills"}
    }
    payload = {'data': json.dumps(payload_dict)}

    try:
        with open(img_path, 'rb') as img_file:
            files = [('file', (filename, img_file, 'application/octet-stream'))]
            response = requests.post(API_URL, headers=HEADERS, data=payload, files=files, timeout=60)
        
        if response.status_code in [200, 202]:
            try:
                task_id = response.json().get("task_id")
                if not task_id:
                    print("❌ Failed: No task_id found.\n")
                    return None
                
                # Fetch the result
                return poll_task_status(task_id)
                    
            except json.JSONDecodeError:
                print("❌ Failed: POST API returned invalid JSON.\n")
        else:
            print(f"❌ Failed to submit. HTTP {response.status_code}\n")
            
    except Exception as e:
        print(f"❌ Submission request failed: {str(e)}\n")
        
    return None

def main(input_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"❌ Input path not found: {input_path}")
        return

    # Tracking metrics for the GLOBAL report
    global_correct = 0
    global_total = 0
    global_report_details = []

    # Find all top-level case folders (e.g., TF202437000201)
    case_folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
    
    for case_name in case_folders:
        case_dir = os.path.join(input_path, case_name)
        
        # Create corresponding output case folder (e.g., output_dir/TF202437000201)
        case_output_dir = os.path.join(output_folder, case_name)
        os.makedirs(case_output_dir, exist_ok=True)
        
        # Tracking metrics for THIS SPECIFIC FOLDER
        case_correct = 0
        case_total = 0
        case_image_details = [] # Store breakdown for the per-folder report
        
        # Walk through the subdirectories (which represent the Ground Truth Class)
        for class_dir_name in os.listdir(case_dir):
            class_dir_path = os.path.join(case_dir, class_dir_name)
            
            if not os.path.isdir(class_dir_path):
                continue
                
            ground_truth_class = class_dir_name # e.g., "CS", "BOE", "CI"
            if ground_truth_class.upper() == "OTHERS":
                continue
            
            # Find images inside this class directory
            images = [f for f in os.listdir(class_dir_path) if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
            
            for img_name in images:
                full_img_path = os.path.join(class_dir_path, img_name)
                
                # The JSON will be saved directly in the case folder: output_dir/TF.../image_name.json
                output_json_path = os.path.join(case_output_dir, f"{os.path.splitext(img_name)[0]}.json")
                
                case_total += 1
                global_total += 1
                
                # Check resume logic
                result_json = None
                if os.path.exists(output_json_path):
                    print(f"⏭️ Skipping {img_name} (JSON already exists)")
                    with open(output_json_path, 'r', encoding='utf-8') as f:
                        result_json = json.load(f)
                else:
                    # Process image
                    result_json = process_single_image(full_img_path)
                    if result_json:
                        # Save JSON directly in the case subfolder
                        with open(output_json_path, 'w', encoding='utf-8') as f:
                            json.dump(result_json, f, indent=4)
                
                # Evaluate Prediction
                predicted_class = "UNKNOWN"
                if result_json:
                    try:
                        # Extract prediction using the first key (image name)
                        img_key = list(result_json.keys())[0]
                        predicted_class = result_json[img_key].get("predicted_class", "UNKNOWN")
                        
                        if predicted_class.upper() == ground_truth_class.upper():
                            case_correct += 1
                            global_correct += 1
                            case_image_details.append(f"[✓] {img_name}: Pred={predicted_class} | GT={ground_truth_class}")
                        else:
                            case_image_details.append(f"[X] {img_name}: Pred={predicted_class} | GT={ground_truth_class}")
                            
                    except Exception as e:
                        print(f"⚠️ Error parsing prediction for {img_name}: {e}")
                        case_image_details.append(f"[!] {img_name}: Error parsing JSON")
                else:
                    case_image_details.append(f"[!] {img_name}: API Failed")

        # ================= Generate PER-FOLDER Report =================
        folder_report_path = os.path.join(case_output_dir, f"report_{case_name}.txt")
        
        # Calculate percentage
        case_pct = (case_correct / case_total * 100) if case_total > 0 else 0
        
        with open(folder_report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== CLASSIFICATION REPORT FOR {case_name} ===\n\n")
            f.write(f"Total Images:        {case_total}\n")
            f.write(f"Correct Predictions: {case_correct}\n")
            f.write(f"Accuracy:            {case_pct:.2f}%\n\n")
            f.write("--- Image Breakdown ---\n")
            for detail in case_image_details:
                f.write(detail + "\n")
                
        print(f"   ↳ Generated folder report: {folder_report_path}")
        
        # Add a summary of this folder to the global report details
        global_report_details.append(f"Folder: {case_name:<20} | Correct: {case_correct}/{case_total} | Accuracy: {case_pct:.2f}%")


    # ================= Generate GLOBAL Report =================
    global_report_path = os.path.join(output_folder, "global_classification_report.txt")
    print("\n" + "="*50)
    print("Generating Global Classification Report...")
    
    with open(global_report_path, 'w', encoding='utf-8') as r:
        r.write("=== OVERALL CLASSIFICATION ACCURACY REPORT ===\n\n")
        
        for line in global_report_details:
            r.write(line + "\n")
            
        r.write("\n" + "="*50 + "\n")
        
        global_pct = (global_correct / global_total * 100) if global_total > 0 else 0
        summary = f"OVERALL GLOBAL ACCURACY: {global_correct}/{global_total} ({global_pct:.2f}%)\n"
        print(summary.strip())
        r.write(summary)
            
    print(f"\n✅ All tasks complete! Global report saved to: {global_report_path}")

if __name__ == "__main__":
    main(INPUT_PATH, OUTPUT_FOLDER)