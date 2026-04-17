import os
import json

def filter_directories_by_image_presence(input_json_path, target_search_folders, final_output_json):
    valid_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    # ---------------------------------------------------------
    # STEP 1: Build a master set of all available image names 
    # from the target folders (recursive search)
    # ---------------------------------------------------------
    print("Scanning target folders recursively to build a master list of images...")
    available_image_names = set()
    
    for folder in target_search_folders:
        if not os.path.exists(folder):
            print(f"  ⚠️ Target folder not found: {folder}")
            continue
            
        # os.walk goes through the folder and all its subfolders automatically
        for root, dirs, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    available_image_names.add(file) # We only store the file name, not the full path
                    
    print(f"✅ Master list built: Found {len(available_image_names)} unique images across all target folders.\n")

    # ---------------------------------------------------------
    # STEP 2: Load the JSON directories and check their images
    # ---------------------------------------------------------
    print("Loading input JSON directories...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        source_directories = json.load(f)
        
    fully_matched_directories = []
    
    print("Cross-checking directories against the master list...")
    for directory in source_directories:
        if not os.path.exists(directory):
            continue
            
        # Get all image names in this specific directory
        images_in_dir = [
            f for f in os.listdir(directory) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        
        # If the directory has no images, we skip it
        if not images_in_dir:
            continue
            
        # Check if ALL images in this directory exist in our master set
        all_present = True
        for img_name in images_in_dir:
            if img_name not in available_image_names:
                all_present = False
                break # Stop checking this directory; one missing image is enough to fail it
                
        if all_present:
            fully_matched_directories.append(directory)

    # ---------------------------------------------------------
    # STEP 3: Print results and save to JSON
    # ---------------------------------------------------------
    print("-" * 50)
    print("=== FINAL ANALYSIS RESULTS ===")
    print(f"Total directories checked: {len(source_directories)}")
    print(f"Directories where ALL images exist in target folders: {len(fully_matched_directories)}")
    print("-" * 50)

    # Save to final JSON
    with open(final_output_json, 'w', encoding='utf-8') as f:
        json.dump(fully_matched_directories, f, indent=4)
        
    print(f"\n✅ Results successfully saved to: {final_output_json}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # 1. The JSON file containing the leaf directories (from your previous script)
    INPUT_JSON_DIRECTORIES = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/dirs_under_30.json"
    
    # 2. The list of root folders where the script should look for the images
    # It will search these folders and all subfolders inside them
    TARGET_SEARCH_FOLDERS = [
        "/datadrive2/IDF_AL_MASRAF/Al_MASHRAF_512_categorized_data",
        "/datadrive2/IDF_AL_MASRAF/Al_MASHRAF_523_categorized_data",
        "/datadrive2/IDF_AL_MASRAF/Al_MASRAF_seocond_set_first_500_categorized_data",
        "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/Al_Masraf_second_set_remaining_500_categorized_data"
    ]
    
    # 3. Where to save the final list of matched directories
    FINAL_OUTPUT_JSON = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/FINAL_MATCHED_DIRECTORIES.json"
    
    filter_directories_by_image_presence(
        input_json_path=INPUT_JSON_DIRECTORIES, 
        target_search_folders=TARGET_SEARCH_FOLDERS, 
        final_output_json=FINAL_OUTPUT_JSON
    )