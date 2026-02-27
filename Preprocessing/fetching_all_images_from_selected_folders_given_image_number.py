'''this code is to fetch all the images from the selected folders 
given the image number. The folders are selected based on the exclusion 
json file and the image number is determined by the target image count. 
The code will ensure that the images are collected proportionally from each source and that 
the final manifest is saved in a json file.'''


import json
import os
import random
import re
from collections import defaultdict

def natural_sort_key(s):
    """Ensures page_2 comes before page_10 in the final list."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def generate_random_manifest(input_jsons, exclusion_json, target_image_count, output_path):
    # 1. Load Exclusion Folder Names
    with open(exclusion_json, 'r') as f:
        exclusion_data = json.load(f)
    # The keys in your exclusion JSON are the folder IDs (leaf names)
    excluded_folders = set(exclusion_data.keys())
    print(f"Loaded {len(excluded_folders)} folders to exclude.")

    manifest_sources = []
    
    # 2. Group images by leaf folder
    for meta_path in input_jsons:
        with open(meta_path, 'r') as f:
            paths = json.load(f)
        
        folder_map = defaultdict(list)
        for p in paths:
            # Captures the immediate parent (the leaf folder)
            leaf_folder = os.path.basename(os.path.dirname(p))
            folder_map[leaf_folder].append(p)
        
        # Filter: Not excluded AND size < 40
        filtered_map = {
            f_name: f_paths for f_name, f_paths in folder_map.items()
            if f_name not in excluded_folders and len(f_paths) < 40
        }
        manifest_sources.append(filtered_map)
        print(f"Source: {os.path.basename(meta_path)} | Eligible Leaf-Folders: {len(filtered_map)}")

    # 3. Proportional Sampling
    final_manifest = []
    target_per_source = target_image_count // len(input_jsons)
    
    for i, source in enumerate(manifest_sources):
        source_collected = 0
        available_folders = list(source.keys())
        random.shuffle(available_folders)
        
        for folder in available_folders:
            if source_collected >= target_per_source:
                break
            
            # Get paths and sort them naturally (page 1, 2, 3...)
            folder_paths = source[folder]
            folder_paths.sort(key=natural_sort_key)
            
            final_manifest.extend(folder_paths)
            source_collected += len(folder_paths)
        
        print(f"Source {i+1} Result: Selected {source_collected} images.")

    # 4. Final Save
    with open(output_path, 'w') as f:
        json.dump(final_manifest, f, indent=4)

    print("-" * 40)
    print(f"TOTAL IMAGES COLLECTED: {len(final_manifest)}")
    print(f"SAVED TO: {output_path}")

if __name__ == "__main__":
    INPUTS = [
        "/datadrive2/IDF_AL_MASRAF/Import_LC_Acceptance_manifest.json",
        "/datadrive2/IDF_AL_MASRAF/Import_LC_Drawing_manifest.json",
        "/datadrive2/IDF_AL_MASRAF/Import_LC_Settlement_manifest.json"
    ]
    EXCLUSION = "/datadrive2/IDF_AL_MASRAF/targeted_sibling_expansion_nested.json"
    generate_random_manifest(INPUTS, EXCLUSION, 1000, "/datadrive2/IDF_AL_MASRAF/final_balanced_sample.json")