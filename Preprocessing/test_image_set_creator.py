'''This script is a Diverse Proportional Sampler. It is designed to create a representative subset of a large document dataset by ensuring two things: page diversity (within a document) and source balance (between categories).
Core Functionality

    Folder-Based Grouping: It treats each parent directory as a single unique document. Instead of just picking the first page, it uses random.choice(images) to select a random page from every folder. This ensures your model sees a mix of covers, middle pages, and signatures rather than just "Page 1."

    Manifest Balancing (Round-Robin): If you provide three different manifests (Acceptance, Settlement, Drawing), the code picks one image from the first, then one from the second, then one from the third. This "Round-Robin" selection ensures that even if one category is much larger than the others, your final 100-image sample is perfectly balanced across all three.

    Randomized Order: By shuffling the unique images before the selection process, the script avoids any bias related to file creation dates or folder names.

Why this is useful for your workflow

    Edge Case Discovery: By picking random pages instead of just the first page, you are more likely to find stamps, handwritten notes, or secondary tables that "Page 1" logic would miss.

    Balanced Training/Testing: It prevents a single category from dominating your sample, which is critical for verifying that your classification rules work across all trade finance types equally.

    Manifest Cleanup: It automatically handles missing files or empty manifests without crashing. '''

import json
import os
import random
from collections import defaultdict

def sample_images_from_manifests(json_paths, total_target=100):
    """
    Takes multiple manifest JSONs, groups images by their parent folder,
    and picks ONE RANDOM image per folder to ensure page diversity.
    """
    manifest_data = []
    
    for path in json_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            all_paths = json.load(f)
            
        # Grouping by directory
        folder_groups = defaultdict(list)
        for img_path in all_paths:
            parent_dir = os.path.dirname(img_path)
            folder_groups[parent_dir].append(img_path)
        
        # --- NEW LOGIC: Pick a random page from each folder ---
        unique_folder_images = []
        for folder, images in folder_groups.items():
            # Instead of imgs[0], we pick a random image from the list
            # This ensures we get a mix of page_1, page_2, etc.
            unique_folder_images.append(random.choice(images))
            
        # Shuffle the list for this manifest so we don't process them in folder order
        random.shuffle(unique_folder_images)
        manifest_data.append(unique_folder_images)

    # 2. Determine sampling balance
    num_manifests = len(manifest_data)
    if num_manifests == 0:
        print("No data found in provided JSONs.")
        return

    final_selection = []
    
    # 3. Round-Robin selection to maintain ratio between manifests
    while len(final_selection) < total_target:
        added_in_this_round = False
        for i in range(num_manifests):
            if len(final_selection) >= total_target:
                break
            
            if manifest_data[i]: 
                # Pop the first random folder image
                final_selection.append(manifest_data[i].pop(0))
                added_in_this_round = True
        
        if not added_in_this_round:
            break

    # 4. Save the balanced manifest
    output_filename = "balanced_diverse_pages_100.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_selection, f, indent=4)

    print("-" * 40)
    print(f"DONE!")
    print(f"Total Unique Images Collected: {len(final_selection)}")
    print(f"Saved to: {output_filename}")
    print("-" * 40)

if __name__ == "__main__":
    JSON_INPUTS = [
        "/datadrive2/IDF_AL_MASRAF/Import_LC_Acceptance_manifest.json",
        "/datadrive2/IDF_AL_MASRAF/Import_LC_Settlement_manifest.json",
        "/datadrive2/IDF_AL_MASRAF/Import_LC_Drawing_manifest.json"
    ]

    sample_images_from_manifests(JSON_INPUTS, total_target=100)