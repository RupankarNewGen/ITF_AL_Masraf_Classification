import logging
import sys
import base64
import io
import json
import secrets
import string
from pathlib import Path
import os

import requests
from datetime import datetime

# ANSI color codes for timing output
RED = '\033[91m'
RESET = '\033[0m'


def get_config(image):
    time1 = datetime.now()
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    base64_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    time2 = datetime.now()
    print(f"{RED}Image base64 encoding time: {time2 - time1}{RESET}")

    time_start = datetime.now()


    """
    ABBYY CONFIG FILE PATH SHOULD UPDATE HERE ...................
    """
    # config_path = "/home/ntlpt52/Projects/ABBYY_/abbyy_config.json"
    # config_path = "/home/ntlpt52/Projects/ABBYY_/copy_abbyy_config.json"
    # config_path = "/home/ntlpt52/Projects/ABBYY_/web_server_abbyy_config.json"
    config_path = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/OCR/abbyy_config.json"



    config_file = json.loads(Path(config_path).read_text())

    config_file["InputFiles"][0]["FileData"]["FileContents"] = base64_image
    config_file["InputFiles"][0]["FileData"]["FileName"] = "dummy_file_name.png"

    return config_file, time_start


def get_job_id(image):
    # This method fetch the job id of abby process which is used to generate end point for further api requests
    config_file, time_start = get_config(image)

    print(f"{RED}Config load + image inject time: {datetime.now() - time_start}{RESET}")
    main_response = requests.post(
        "http://10.2.2.10:8080/FineReaderServer14/api/workflows/Default%20Workflow/input/ticket",
        headers={"Content-Type": "application/json"},
        data=json.dumps(config_file),
        verify=False)

    print(f"{RED}Job submission time: {datetime.now() - time_start}{RESET}")


    # main_response = requests.post(
    #     ocr_constants.MAIN_RESPONSE_STR.format(self._config[ocr_constants.ABBY_API],
    #                                             self._config[ocr_constants.ABBY_PROCESSING_ENDPOINT]),
    #     headers={ocr_constants.CONTENT_TYPE_KEY: self._config[ocr_constants.CONTENT_TYPE_KEY]},
    #     data=json.dumps(config_file),
    #     verify=self.ssl_verification)


    try:
        job_id = main_response.text.replace("{", "").replace("}", "").replace('"', '')
        print("\n Job ID: ", job_id,"\n")

    except Exception as e:
        raise RuntimeError(f"Exception is ===> {e}\nResponse is ===> {main_response.text}")
    return job_id



def calculate_ocr_single(image):
    job_id = get_job_id(image)


    ocr_exec_start_time = datetime.now()
    api_status_and_result_endpoint = "http://10.2.2.10:8080/FineReaderServer14/api/jobs/%7B{}%7D".format(job_id)
    while True:
        status_response = requests.get(api_status_and_result_endpoint, verify=False)

        # print("API status response: ", status_response)
        # print("API status response: ", status_response.text)

        state_info = status_response.json()
        if state_info["State"] == "JS_Complete":
            break
        elif state_info["State"] == "JS_NosuchJob":
            raise KeyError("Job \"%s\" not found" % job_id)

    print(f"{RED}OCR processing time: {datetime.now() - ocr_exec_start_time}{RESET}")
    text_result = None
    # text_result = requests.get(
    #     "http://10.2.2.10:8080/FineReaderServer14/api/jobs/%7B{}%7D/result/outputDocuments/1/files/0".format(job_id),
    #     verify=False)

    result_fetch_start = datetime.now()
    json_result = requests.get(
        "http://10.2.2.10:8080/FineReaderServer14/api/jobs/%7B{}%7D/result/outputDocuments/2/files/0".format(job_id),
        verify=False)
    print(f"{RED}JSON result fetch time: {datetime.now() - result_fetch_start}{RESET}")
    print(f"{RED}JSON result status code: {json_result.status_code}{RESET}")
    print(f"{RED}JSON result content (first 500 chars): {json_result.text[:500]}{RESET}")

    if json_result.status_code != 200 or not json_result.text.strip():
        raise RuntimeError(
            f"Failed to fetch JSON result. Status: {json_result.status_code}, "
            f"Body: {json_result.text[:500]}"
        )

    print(f"{RED}Total OCR pipeline time: {datetime.now() - ocr_exec_start_time}{RESET}")
    return text_result, json_result.json()



import glob
from PIL import Image


def transform_abbyy_to_simplified(abbyy_json):
    """
    Transform ABBYY FineReaderServer JSON output to simplified format.
    
    Input: ABBYY complex nested JSON structure
    Output: {'ocrContent': [{'word': str, 'left': int, 'top': int, 'width': int, 'height': int, 'x1': int, 'y1': int, 'x2': int, 'y2': int}, ...]}
    """
    ocr_content = []
    
    try:
        # Navigate through ABBYY structure: layout -> pages -> texts -> words
        if 'layout' not in abbyy_json or 'pages' not in abbyy_json['layout']:
            return {'ocrContent': []}
        
        pages = abbyy_json['layout']['pages']
        
        for page in pages:
            if 'texts' not in page:
                continue
            
            texts = page['texts']
            
            for text_block in texts:
                if 'lines' not in text_block:
                    continue
                
                lines = text_block['lines']
                
                for line in lines:
                    if 'words' not in line:
                        continue
                    
                    words = line['words']
                    
                    for word_obj in words:
                        word_text = word_obj.get('text', '')
                        
                        # Extract position information
                        if 'position' in word_obj:
                            pos = word_obj['position']
                            left = pos.get('l', 0)
                            top = pos.get('t', 0)
                            right = pos.get('r', 0)
                            bottom = pos.get('b', 0)
                            
                            width = right - left
                            height = bottom - top
                            
                            word_entry = {
                                'word': word_text,
                                'left': left,
                                'top': top,
                                'width': width,
                                'height': height,
                                'x1': left,
                                'y1': top,
                                'x2': right,
                                'y2': bottom
                            }
                            
                            ocr_content.append(word_entry)
    
    except Exception as e:
        print(f"Warning: Error transforming ABBYY JSON: {e}")
    
    return {'ocrContent': ocr_content}


# ================= CONFIGURATION =================
# List of input folders to process (searches recursively in each)
INPUT_FOLDERS = [
    "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/test_set2",
    "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/test_set_1",
    "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/DOC_3_splitted",
    "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/DOC_5_splitted",
]

# Output folder for all results
OUTPUT_FOLDER = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Test_ocr_rupankar"

# ================= END CONFIGURATION =================

def find_images_recursive(folder_path):
    """
    Recursively find all image files in a folder.
    Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
    """
    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                       '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF')
    
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    
    return sorted(image_files)

def main():
    """
    Process all input folders and save results to output folder.
    """
    print("=" * 60)
    print("ABBYY OCR Batch Processing")
    print("=" * 60)
    print(f"Output folder: {OUTPUT_FOLDER}\n")
    
    total_processed = 0
    total_failed = 0
    
    for input_folder in INPUT_FOLDERS:
        if not os.path.exists(input_folder):
            print(f"⚠️  Skipping (folder not found): {input_folder}\n")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing folder: {input_folder}")
        print(f"{'='*60}")
        
        # Find images in this folder
        image_files = find_images_recursive(input_folder)
        
        if not image_files:
            print(f"✓ No images found in {input_folder}\n")
            continue
        
        print(f"✅ Found {len(image_files)} images to process\n")
        
        processed_count = 0
        failed_count = 0
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
                
                # Load image
                image = Image.open(image_path)
                
                # Process with OCR
                text, json_data = calculate_ocr_single(image)
                
                # Generate output filename: {image_name}_textAndCoordinates.txt
                image_basename = os.path.basename(image_path)
                output_filename = os.path.splitext(image_basename)[0] + "_textAndCoordinates.txt"
                output_file_path = os.path.join(OUTPUT_FOLDER, output_filename)
                
                # Transform ABBYY format to simplified format
                simplified_ocr_data = transform_abbyy_to_simplified(json_data)
                
                # Save as TXT file (convert dict to string representation)
                with open(output_file_path, "w", encoding='utf-8') as f:
                    f.write(str(simplified_ocr_data))
                
                print(f"  ✅ Saved: {output_filename}\n")
                processed_count += 1
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}\n")
                failed_count += 1
                continue
        
        # Folder summary
        print("-" * 60)
        print(f"Folder Complete: {os.path.basename(input_folder)}")
        print(f"  Processed: {processed_count}")
        print(f"  Failed: {failed_count}")
        print("-" * 60)
        
        total_processed += processed_count
        total_failed += failed_count
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total folders processed: {len([f for f in INPUT_FOLDERS if os.path.exists(f)])}")
    print(f"Total images processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 