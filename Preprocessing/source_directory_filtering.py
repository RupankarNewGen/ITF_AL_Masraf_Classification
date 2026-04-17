import os
import json

def filter_directories(input_json_path, output_under_30_json, output_fully_in_json):
    print("Loading input JSON...")
    
    # 1. Load the input paths
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_paths = json.load(f)
        
    # Convert list to a set for lightning-fast lookups later
    input_paths_set = set(input_paths)
    
    # 2. Extract unique leaf directories
    # os.path.dirname gets the folder path just before the file name
    leaf_dirs = set(os.path.dirname(path) for path in input_paths)
    print(f"Found {len(leaf_dirs)} unique leaf directories in the input JSON.\n")
    
    # Valid image extensions to look for on the OS
    valid_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    dirs_under_30 = []
    dirs_fully_in_json = []
    
    print("Scanning physical directories on disk...")
    # 3. Analyze each directory
    for d in leaf_dirs:
        if not os.path.exists(d):
            print(f"  ⚠️ Directory not found on disk, skipping: {d}")
            continue
            
        # Collect all actual image files present in the directory on the OS
        actual_images_on_disk = []
        for filename in os.listdir(d):
            if os.path.splitext(filename)[1].lower() in valid_exts:
                # Reconstruct the full path to match the input format
                full_path = os.path.join(d, filename)
                actual_images_on_disk.append(full_path)
                
        img_count = len(actual_images_on_disk)
        
        # TASK 1: Filter directories with <= 30 images
        if 0 < img_count <= 30:
            dirs_under_30.append(d)
            
            # TASK 2: Check if ALL physical images in this folder exist in the input JSON
            all_accounted_for = True
            for img_path in actual_images_on_disk:
                if img_path not in input_paths_set:
                    all_accounted_for = False
                    break # Stop checking this folder as soon as we find one missing image
            
            if all_accounted_for:
                dirs_fully_in_json.append(d)

    # 4. Print the final counts
    print("-" * 50)
    print("=== ANALYSIS RESULTS ===")
    print(f"1. Directories with <= 30 images: {len(dirs_under_30)}")
    print(f"2. Directories where ALL images are in the JSON: {len(dirs_fully_in_json)}")
    print("-" * 50)

    # 5. Save the results to the output JSON files
    with open(output_under_30_json, 'w', encoding='utf-8') as f:
        json.dump(dirs_under_30, f, indent=4)
        
    with open(output_fully_in_json, 'w', encoding='utf-8') as f:
        json.dump(dirs_fully_in_json, f, indent=4)

    print(f"\n✅ Results saved to:\n  - {output_under_30_json}\n  - {output_fully_in_json}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # The JSON file you provided containing the list of image paths
    INPUT_JSON = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/2000_images_set_march_5_2026.json" 
    
    # Where to save the output lists
    OUTPUT_1_UNDER_30 = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/dirs_under_30.json"
    OUTPUT_2_FULLY_IN_JSON = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/dirs_fully_in_json.json"
    
    filter_directories(INPUT_JSON, OUTPUT_1_UNDER_30, OUTPUT_2_FULLY_IN_JSON)