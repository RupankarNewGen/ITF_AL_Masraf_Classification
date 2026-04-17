import os
import json

def find_full_image_paths(target_folders, master_json_path, output_json_path):
    # Standard image extensions to ensure we only look at images
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    
    # ---------------------------------------------------------
    # STEP 1: Scan target folders and collect image file names
    # ---------------------------------------------------------
    print("Scanning target folders for image names...")
    target_image_names = set()
    
    for folder in target_folders:
        if not os.path.exists(folder):
            print(f"  ⚠️ Folder not found, skipping: {folder}")
            continue
            
        # os.walk allows us to search recursively if there are subfolders
        for root, _, files in os.walk(folder):
            for filename in files:
                # Check if it's a valid image file
                if os.path.splitext(filename)[1].lower() in valid_exts:
                    target_image_names.add(filename)
                    
    print(f"✅ Found {len(target_image_names)} unique image names in your physical folders.\n")

    # ---------------------------------------------------------
    # STEP 2: Load the Master JSON (the superset of full paths)
    # ---------------------------------------------------------
    print(f"Loading Master JSON: {os.path.basename(master_json_path)}...")
    try:
        with open(master_json_path, 'r', encoding='utf-8') as f:
            master_paths = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load Master JSON: {e}")
        return

    print(f"✅ Loaded {len(master_paths)} total paths from the Master JSON.\n")

    # ---------------------------------------------------------
    # STEP 3: Match file names to full paths
    # ---------------------------------------------------------
    print("Cross-referencing file names to find full paths...")
    matched_full_paths = []
    
    for full_path in master_paths:
        # Extract just the file name from the long absolute path
        filename_in_json = os.path.basename(full_path)
        
        # If this file name is in the set we collected from the physical folders, save the full path
        if filename_in_json in target_image_names:
            matched_full_paths.append(full_path)

    # ---------------------------------------------------------
    # STEP 4: Save the matched paths to the new output JSON
    # ---------------------------------------------------------
    print(f"Saving {len(matched_full_paths)} matched full paths to output JSON...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(matched_full_paths, f, indent=4)
        
    print("-" * 50)
    print("=== SUMMARY ===")
    print(f"Images requested:   {len(target_image_names)}")
    print(f"Full paths found:   {len(matched_full_paths)}")
    print(f"Results saved to:   {output_json_path}")
    print("-" * 50)


if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    
    # 1. The list of your physical folders containing the actual image files
    TARGET_FOLDERS = [
        "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/addtional_images_similar_to_2000",
        "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/remianing_images_to_reach_10"
    ]
    
    # 2. The JSON file containing the superset list of full absolute paths
    MASTER_JSON_PATH = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/Import_LC_Drawing_manifest.json"
    
    # 3. Where you want the final list of matched full paths to be saved
    OUTPUT_JSON_PATH = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/lcdrawing_images_already_used_in_traning"
    
    find_full_image_paths(
        target_folders=TARGET_FOLDERS,
        master_json_path=MASTER_JSON_PATH,
        output_json_path=OUTPUT_JSON_PATH
    )