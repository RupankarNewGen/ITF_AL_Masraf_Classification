import os
import json
import shutil
from tqdm import tqdm

def collect_images_to_leaf_folders(json_paths, output_root):
    """
    Copies images from JSON manifests into the output_root, 
    creating subfolders named after the image's original leaf directory.
    """
    # Ensure the main destination root exists
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    all_image_paths = []
    
    # Load all paths from the provided JSON manifests
    for json_path in json_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                paths = json.load(f)
                all_image_paths.extend(paths)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    print(f"Total images to process: {len(all_image_paths)}")

    copied_count = 0
    # Use a set to keep track of created directories to avoid redundant os.path checks
    created_dirs = set()

    for img_path in tqdm(all_image_paths, desc="Organizing into Leaf Folders"):
        if not os.path.exists(img_path):
            continue
            
        # 1. Extract filename and the immediate parent (leaf) folder name
        filename = os.path.basename(img_path)
        leaf_folder = os.path.basename(os.path.dirname(img_path))
        
        # 2. Construct the target path: output_root/leaf_folder/filename
        target_dir = os.path.join(output_root, leaf_folder)
        
        if target_dir not in created_dirs:
            os.makedirs(target_dir, exist_ok=True)
            created_dirs.add(target_dir)

        dest_path = os.path.join(target_dir, filename)

        # 3. Copy the file
        try:
            shutil.copy2(img_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"\nFailed to copy {filename}: {e}")

    print("-" * 50)
    print(f"PROCESS COMPLETED")
    print(f"Output Root: {output_root}")
    print(f"Total Subfolders (Leafs): {len(created_dirs)}")
    print(f"Total Images Copied:      {copied_count}")
    print("-" * 50)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Input JSON containing the full image paths
    MANIFEST_JSONS = [
        "/datadrive2/IDF_AL_MASRAF/remaining_untapped_images.json"
    ]
    
    # The main directory where subfolders will be created
    DESTINATION_ROOT = "/datadrive2/IDF_AL_MASRAF/remaining_untapped_images_512"

    collect_images_to_leaf_folders(MANIFEST_JSONS, DESTINATION_ROOT)