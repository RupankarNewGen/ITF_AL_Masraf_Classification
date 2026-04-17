import os
import json
import time
import requests

# ================= CONFIGURATION =================

# 1. The POST API URL (to submit the image)
#API_URL = "http://10.2.0.35:8021/bundleAPI/ntdemoinstance_outlook_com_1773831699-56428b2d-f969-4a02-97f9-2e5766bdf1b1/V3/1774334159680"
API_URL = "http://10.2.0.36:8021/bundleAPI/ntdemoinstance_outlook_com_1773831699-56428b2d-f969-4a02-97f9-2e5766bdf1b1/V3/1774334159680"

# 2. The GET API URL Base (to check the status)
#STATUS_API_BASE_URL = "http://10.2.0.35:8021/task-status/"
STATUS_API_BASE_URL = "http://10.2.0.36:8021/task-status/"

# 3. Input and Output Paths
# You can set INPUT_PATH to EITHER a single file OR a folder directory.
INPUT_PATH = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/addtional_images_similar_to_2000/BOE/ABINAEAA_1/Trade Finance_20210518171838_1.00_page_10.jpeg"
OUTPUT_FOLDER = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/test"

# Authorization Header
HEADERS = {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhdXRoZW50aWNhdGVkX3VzZXIiLCJleHAiOjE3OTU5MTQyMTR9.vka3MnQGXe2qeOAOico0AtQsXEqZNUqVAzaJPbt5a-s'
}

# Valid image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Polling settings
POLL_INTERVAL_SECONDS = 5
MAX_RETRIES = 60  # 60 retries * 5 seconds = 5 minutes max wait time per image

# =================================================


def poll_task_status(task_id, output_json_path):
    """Loops and checks the status API until the task is complete."""
    status_url = f"{STATUS_API_BASE_URL}{task_id}"
    print(f"   ↳ Task ID received: {task_id}")
    print("   ↳ Polling for results", end="", flush=True)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(status_url, headers=HEADERS)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "").upper()
                
                # If status is PENDING or PROCESSING, keep waiting
                if status in ["PENDING", "PROCESSING", "STARTED"]:
                    print(".", end="", flush=True)
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
                
                # If status indicates success or result is populated
                elif status in ["SUCCESS", "COMPLETED", "DONE"] or data.get("result") is not None:
                    print(" ✓ Done!")
                    
                    # Extract the actual result payload
                    final_result = data.get("result", data)
                    
                    # Save to the output file
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(final_result, f, indent=4)
                    
                    print(f"   ↳ Saved to: {os.path.basename(output_json_path)}\n")
                    return True
                
                # If status indicates a failure
                elif status in ["FAILED", "FAILURE", "ERROR"]:
                    print(f" ❌ Server failed to process the image. Status: {status}\n")
                    return False
                
                else:
                    print(f" ⚠️ Unknown status received: {status}\n")
                    return False
                    
            else:
                print(f" ❌ Error checking status. HTTP {response.status_code}\n")
                return False
                
        except Exception as e:
            print(f" ❌ Polling request failed: {str(e)}\n")
            return False

    print(" ❌ Timeout reached! The server took too long to process this image.\n")
    return False


def process_single_image(img_path, output_folder):
    """Submits a single image to the API and triggers the polling loop."""
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    output_json_path = os.path.join(output_folder, f"{base_name}.json")

    # Resume Logic
    if os.path.exists(output_json_path):
        print(f"⏭️ Skipping {filename} (JSON already exists)")
        return

    print(f"Processing: {filename}")

    # 1. Prepare dynamic payload
    payload_dict = {
        "fileType": "img",
        "image_name": filename,
        "ocr_flag": "false",
        "other_params": {
            "product_name": "Import Bills"
        }
    }
    
    payload = {'data': json.dumps(payload_dict)}

    # 2. Make the POST request to submit the image
    try:
        with open(img_path, 'rb') as img_file:
            files = [('file', (filename, img_file, 'application/octet-stream'))]
            response = requests.post(API_URL, headers=HEADERS, data=payload, files=files)
        
        # 3. Handle the submission response
        if response.status_code in [200, 202]:
            try:
                response_json = response.json()
                task_id = response_json.get("task_id")
                
                if not task_id:
                    print("❌ Failed: Server responded with success but no task_id was found.\n")
                    return
                
                # 4. Trigger the polling loop
                poll_task_status(task_id, output_json_path)
                
            except json.JSONDecodeError:
                print("❌ Failed: POST API returned invalid JSON.\n")
        else:
            print(f"❌ Failed to submit. HTTP {response.status_code}: {response.text[:100]}\n")
            
    except Exception as e:
        print(f"❌ Submission request failed: {str(e)}\n")


def main(input_path, output_folder):
    """Determines if the input is a file or a folder and routes accordingly."""
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"❌ Input path not found: {input_path}")
        return

    # Route 1: It's a single file
    if os.path.isfile(input_path):
        if os.path.splitext(input_path)[1].lower() in VALID_EXTENSIONS:
            print("🎯 Single image detected.")
            process_single_image(input_path, output_folder)
        else:
            print("❌ The file provided is not a valid image format.")
            
    # Route 2: It's a folder
    elif os.path.isdir(input_path):
        image_files = [f for f in os.listdir(input_path) 
                       if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
        
        if not image_files:
            print(f"⚠️ No valid images found in folder: {input_path}")
            return
            
        print(f"🎯 Folder detected. Found {len(image_files)} images to process.\n")
        
        # Process them alphabetically/sequentially
        for filename in sorted(image_files):
            full_img_path = os.path.join(input_path, filename)
            process_single_image(full_img_path, output_folder)

    print("✅ All tasks complete!")


if __name__ == "__main__":
    main(INPUT_PATH, OUTPUT_FOLDER)