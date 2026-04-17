import os
import json
import shutil
from tqdm import tqdm

def organize_images_by_class(json_folder, image_source_folder, main_output_folder):
    if not os.path.exists(main_output_folder):
        os.makedirs(main_output_folder)

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    moved_count = 0
    error_count = 0

    # PRE-SCAN: Create a map of filename -> full_path to avoid slow recursive searches inside the loop
    print("Indexing source images... (Scanning subfolders)")
    image_map = {}
    for root, _, files in os.walk(image_source_folder):
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_extensions:
                image_map[name] = os.path.join(root, f)

    for json_filename in tqdm(json_files, desc="Organizing Images"):
        base_name = os.path.splitext(json_filename)[0]
        json_path = os.path.join(json_folder, json_filename)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                class_name = data.get("classification", "others")

            # 1. Create class-specific subfolder
            class_folder_path = os.path.join(main_output_folder, class_name)
            os.makedirs(class_folder_path, exist_ok=True)

            # 2. Find matching image from our pre-scanned map
            if base_name in image_map:
                source_path = image_map[base_name]
                filename = os.path.basename(source_path)
                dest_path = os.path.join(class_folder_path, filename)
                
                shutil.copy2(source_path, dest_path)
                moved_count += 1
            else:
                print(f"Warning: Image {base_name} not found anywhere in {image_source_folder}")

        except Exception as e:
            print(f"Error processing {json_filename}: {e}")
            error_count += 1

    print(f"\nORGANIZATION COMPLETE. Images Organized: {moved_count}")

if __name__ == "__main__":
    JSON_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/output_jsons/532_first_set_instruct_model_result"
    # POINT THIS TO THE MAIN FOLDER CONTAINING THE SUBFOLDERS
    IMAGE_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_input_Data/first_set_images_532"
    FINAL_OUTPUT_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/532_images_organized_by_predictedclass"

    organize_images_by_class(JSON_DIR, IMAGE_DIR, FINAL_OUTPUT_DIR)