import os
import json
import shutil
import argparse

def organize_random_directories(input_directory, target_folders, main_output_folder):
    valid_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    # ---------------------------------------------------------
    # STEP 1: Define the mapping from long names to standard short names
    # ---------------------------------------------------------
    class_mapping = {
        "Bill_of_Exchange": "BOE", "BOE": "BOE",
        "Bill_of_Lading": "BOL", "BOL": "BOL",
        "Commercial_Invoice": "CI", "CI": "CI",
        "Certificate_of_Origin": "COO", "COO": "COO",
        "Covering_Schedule": "CS", "CS": "CS",
        "Packing_List": "PL", "PL": "PL",
        "Others": "Others", "others": "Others"
    }

    # ---------------------------------------------------------
    # STEP 2: Pre-scan Target Folders to map image_name -> class_name
    # ---------------------------------------------------------
    print("Scanning target folders to learn image classifications...")
    image_to_class_map = {}
    
    for target_root in target_folders:
        if not os.path.exists(target_root):
            print(f"  ⚠️ Target folder not found: {target_root}")
            continue
            
        # Look only at the 1st level of subdirectories
        for subdir_name in os.listdir(target_root):
            subdir_path = os.path.join(target_root, subdir_name)
            
            if os.path.isdir(subdir_path) and subdir_name in class_mapping:
                standard_class_name = class_mapping[subdir_name]
                
                # Register all images in this subdirectory
                for img_file in os.listdir(subdir_path):
                    if os.path.splitext(img_file)[1].lower() in valid_exts:
                        image_to_class_map[img_file] = standard_class_name

    print(f"✅ Learned classifications for {len(image_to_class_map)} unique images.\n")

    # ---------------------------------------------------------
    # STEP 3: Process and organize the selected directory
    # ---------------------------------------------------------
    os.makedirs(main_output_folder, exist_ok=True)
    
    print("\nCopying and organizing images...")
    
    if not os.path.exists(input_directory):
        print(f"⚠️ Input directory not found: {input_directory}")
        return

    # The name of the directory itself (e.g., TF203077000001)
    dir_base_name = os.path.basename(os.path.normpath(input_directory))
    output_dir_path = os.path.join(main_output_folder, dir_base_name)
    
    # Get images in the source directory
    images = [f for f in os.listdir(input_directory) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not images:
        print(f"  ⚠️ No images found in {input_directory}")
        return
        
    for img_name in images:
        src_img_path = os.path.join(input_directory, img_name)
        
        # Look up its class (fallback to 'Others' if somehow missing)
        standard_class = image_to_class_map.get(img_name, "Others")
        
        # Create the specific class folder inside this directory's output folder
        class_folder_path = os.path.join(output_dir_path, standard_class)
        os.makedirs(class_folder_path, exist_ok=True)
        
        # Copy the file
        dest_img_path = os.path.join(class_folder_path, img_name)
        shutil.copy2(src_img_path, dest_img_path)
        
    print(f"  ✓ Processed: {dir_base_name} ({len(images)} images)")

    print(f"\n✅ All done! Organized files are in: {main_output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize categorised images for a single directory.")
    parser.add_argument("--directory", required=True, help="The single input directory to process")
    args = parser.parse_args()
    
    # --- CONFIGURATION ---
    
    # The folders where the categorized images are stored
    TARGET_SEARCH_FOLDERS = [
        "/datadrive2/IDF_AL_MASRAF/Al_MASHRAF_512_categorized_data",
        "/datadrive2/IDF_AL_MASRAF/Al_MASHRAF_523_categorized_data",
        "/datadrive2/IDF_AL_MASRAF/Al_MASRAF_seocond_set_first_500_categorized_data",
        "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/Al_Masraf_second_set_remaining_500_categorized_data"
    ]
    
    # Where you want the newly organized folder to be saved
    MAIN_OUTPUT_FOLDER = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/images_for_inference"
    
    organize_random_directories(
        input_directory=args.directory, 
        target_folders=TARGET_SEARCH_FOLDERS, 
        main_output_folder=MAIN_OUTPUT_FOLDER
    )