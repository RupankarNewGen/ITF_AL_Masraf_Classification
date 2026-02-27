'''This script is a Missing File Auditor and Manifest Generator. It is designed to identify "untapped" data by comparing a list of already processed files against the actual physical files stored on your disk.
Core Functionality

    Set-Based Comparison: It loads your current manifest (targeted_sibling_expansion.json) into a Python set. This allows for extremely fast (O(1)) lookups to check if a file has already been indexed, even if you have thousands of images.

    Parent Directory Targeting: Instead of scanning your entire server, it intelligently extracts the unique parent directories from your existing manifest. It then limits its search only to these specific folders.

    Difference Detection: The script performs a directory scan and collects only the images that exist on the drive but are not present in your JSON. This effectively finds the "leftover" pages or files in a document set.

    Natural Sorting: It uses a natural sort algorithm to ensure the output list respects numerical order (e.g., page_2.jpg correctly comes before page_10.jpg), which is vital for maintaining document page sequences.

    Manifest Output: The final "leftover" list is saved into a new JSON file (remaining_untapped_images.json), ready to be used as a new processing batch.

Why this is useful for your workflow

    Gap Analysis: It ensures 100% coverage of your document folders by highlighting exactly which images were missed in previous sampling or expansion runs.

    Incremental Batching: It allows you to create "Phase 2" manifests without worrying about duplicating work from "Phase 1."

    Efficiency: By using absolute paths and sets, it handles large-scale trade finance document dumps with minimal memory and CPU overhead. '''


import os
import json
import re

def natural_sort_key(s):
    """Ensures page_2 comes before page_10."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def find_remaining_images(input_json_path, output_json_path):
    if not os.path.exists(input_json_path):
        print(f"Error: {input_json_path} not found.")
        return

    # 1. Load the existing images from your JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        existing_images = json.load(f)
    
    # Convert to a set for O(1) lookup speed
    existing_set = set(os.path.abspath(p) for p in existing_images)

    # 2. Identify the unique parent folders involved
    target_folders = set(os.path.dirname(p) for p in existing_images)
    
    remaining_images = []
    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    print(f"Scanning {len(target_folders)} folders for remaining images...")

    # 3. Scan the folders for files NOT in the existing set
    for folder in target_folders:
        if not os.path.exists(folder):
            continue
            
        # Get all valid images in this specific folder
        disk_files = [os.path.abspath(os.path.join(folder, f)) for f in os.listdir(folder)
                      if os.path.splitext(f)[1].lower() in valid_extensions]
        
        # Filter: Keep only the ones we don't already have
        for file_path in disk_files:
            if file_path not in existing_set:
                remaining_images.append(file_path)

    # 4. Sort the result naturally
    remaining_images.sort(key=natural_sort_key)

    # 5. Save the remaining images to a new JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(remaining_images, f, indent=4)

    # 6. Final Statistics
    print("-" * 50)
    print(f"COMPLETED")
    print(f"Total Folders Scanned:    {len(target_folders)}")
    print(f"Existing Images Skipped:  {len(existing_images)}")
    print(f"Remaining Images Found:   {len(remaining_images)}")
    print(f"New Manifest Saved To:    {output_json_path}")
    print("-" * 50)

if __name__ == "__main__":
    IN_JSON = "/datadrive2/IDF_AL_MASRAF/targeted_sibling_expansion.json"
    OUT_JSON = "/datadrive2/IDF_AL_MASRAF/remaining_untapped_images.json"

    find_remaining_images(IN_JSON, OUT_JSON)