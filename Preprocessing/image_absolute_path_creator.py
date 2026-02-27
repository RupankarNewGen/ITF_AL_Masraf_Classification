'''This script is a recursive manifest generator. It acts as a targeted search tool to find specific document categories across a large directory tree and catalog all associated images into a single JSON file.
Core Functionality

    Case-Insensitive Target Matching: It walks through the ROOT_PATH looking for any folder that matches your TARGET string (e.g., "Import LC Drawing"), regardless of whether the folder name is uppercase, lowercase, or mixed.

    Deep Image Harvesting: Once a matching folder is found, the script performs a secondary recursive search inside that folder to find every image file with standard extensions (.jpg, .png, .tiff, etc.).

    Absolute Path Mapping: It converts every image found into an absolute path (the full system address). This is crucial for machine learning pipelines so that the data loader always knows exactly where the file is located.

    Structured Output: It saves the final list as a clean, formatted JSON file named after your target.

Why this is useful for your workflow

    De-cluttering: Instead of manually searching through thousands of folders, you get a single list of files ready for batch processing.

    Portability: The generated _manifest.json can be moved between scripts (like your classification or extraction scripts) without needing to re-scan the entire drive.

    Consistency: It ensures that every page of every document within that specific trade finance category is captured for processing.'''




import os
import json

def generate_image_manifest(root_path, target_folder_name):
    """
    Scans root_path for 'target_folder_name' (Case-Insensitive). 
    Collects absolute paths of all images found inside it (recursively).
    """
    image_list = []
    # Supported image formats
    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    
    # Standardize target for comparison
    target_lower = target_folder_name.lower().strip()
    
    print(f"--- Searching for: '{target_folder_name}' (Case-Insensitive) ---")
    
    found_folders = []
    
    # Step 1: Walk through the root directory
    for root, dirs, files in os.walk(root_path):
        # Compare current folder name to target (both lowercase)
        if os.path.basename(root).lower() == target_lower:
            found_folders.append(root)
            print(f"Match found at: {root}")
            
            # Step 2: Search recursively inside this folder for images
            for sub_root, sub_dirs, sub_files in os.walk(root):
                for file in sub_files:
                    if os.path.splitext(file)[1].lower() in valid_extensions:
                        abs_path = os.path.abspath(os.path.join(sub_root, file))
                        image_list.append(abs_path)

    if not image_list:
        print(f"\nDone. No images found in any folder matching '{target_folder_name}'")
        return

    # Step 3: Save results to JSON
    # Creating a clean filename
    json_filename = f"{target_folder_name.replace(' ', '_')}_manifest.json"
    
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(image_list, f, indent=4)

    print("-" * 40)
    print(f"SUCCESS!")
    print(f"Total Folders Found: {len(found_folders)}")
    print(f"Total Images Found:  {len(image_list)}")
    print(f"Saved to:            {json_filename}")
    print("-" * 40)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # The starting point for the search
    ROOT_PATH = "/datadrive2/IDF_AL_MASRAF/DocumentDumpImages" 
    
    # Target folder name (Case will be ignored)
    TARGET = "Import LC Drawing" 

    generate_image_manifest(ROOT_PATH, TARGET)