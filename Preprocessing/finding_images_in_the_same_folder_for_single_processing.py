'''This script is a Dynamic Contextual Expander. Its primary purpose is to take a small sample of specific "seed" images and automatically pull in their "siblings" (neighboring pages) from the same folders to reach a target dataset size, ensuring no data is wasted.
Core Functionality

    Target-Folder Identification: The script looks at the TARGET_LIST (filenames of interest) and identifies which parent folders they live in.

    Dynamic "Water-Filling" Allocation: This is the most intelligent part of the code. Instead of simply taking N images per folder, it sets a global goal (e.g., 14×number of folders).

        If a folder is small (e.g., only has 5 images), the script takes all 5.

        It then redistributes the "missing" 9 slots to larger folders that have extra capacity.

        This ensures you hit your exact total_target_count even if some folders are nearly empty.

    Natural Sorting: It uses a natural sort key (1,2,10 instead of 1,10,2) when inventorying folders. This ensures that when images are extracted, they are pulled in their actual chronological page order.

    Inventory Reporting: Before saving, it prints a table showing the capacity of each folder versus how many images were actually allocated, providing full transparency into the expansion process.  '''


import os
import json
import re
from tqdm import tqdm

def natural_sort_key(s):
    """
    Helper function to provide natural sorting (e.g., 1, 2, 10 instead of 1, 10, 2).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def collect_dynamic_siblings(input_json_path, output_json_path, target_filenames, n_avg=25):
    """
    - Targets a total of (n_avg * total_folders) images.
    - If a folder is small, the deficit is distributed to larger folders.
    """
    if not os.path.exists(input_json_path):
        print(f"Error: Input file {input_json_path} not found.")
        return

    # 1. Load the existing manifest
    with open(input_json_path, 'r', encoding='utf-8') as f:
        all_paths = json.load(f)

    # 2. Map target filenames to parent folders
    target_parent_folders = set()
    for full_path in all_paths:
        filename = os.path.basename(full_path)
        if any(target in filename for target in target_filenames):
            parent_dir = os.path.dirname(full_path)
            target_parent_folders.add(parent_dir)

    num_folders = len(target_parent_folders)
    total_target_count = n_avg * num_folders
    print(f"Found {num_folders} folders. Total Target: {total_target_count} images (~{n_avg} per folder).")

    # 3. Analyze folder capacities
    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    folder_inventory = {} # { path: [list_of_files] }

    for folder in target_parent_folders:
        if not os.path.exists(folder): continue
        files = [os.path.abspath(os.path.join(folder, f)) for f in os.listdir(folder) 
                 if os.path.splitext(f)[1].lower() in valid_extensions]
        
        # CHANGED: natural sorting instead of default lexicographical sort
        files.sort(key=natural_sort_key)
        
        folder_inventory[folder] = files

    # 4. DYNAMIC ALLOCATION LOGIC
    current_allocations = {folder: 0 for folder in folder_inventory}
    images_remaining_to_reach_target = total_target_count
    
    while images_remaining_to_reach_target > 0:
        available_folders = [f for f in folder_inventory if current_allocations[f] < len(folder_inventory[f])]
        
        if not available_folders:
            print("Warning: Exhausted all images in all folders before reaching target.")
            break
            
        per_folder_boost = max(1, images_remaining_to_reach_target // len(available_folders))
        
        for folder in available_folders:
            if images_remaining_to_reach_target <= 0: break
            capacity_left = len(folder_inventory[folder]) - current_allocations[folder]
            to_take = min(per_folder_boost, capacity_left, images_remaining_to_reach_target)
            current_allocations[folder] += to_take
            images_remaining_to_reach_target -= to_take

    # 5. Extract the images based on final allocations
    final_image_list = []
    print("-" * 50)
    print(f"{'Folder':<30} | {'Capacity':<10} | {'Allocated':<10}")
    print("-" * 50)
    
    for folder, count in current_allocations.items():
        print(f"{os.path.basename(folder):<30} | {len(folder_inventory[folder]):<10} | {count:<10}")
        final_image_list.extend(folder_inventory[folder][:count])

    # 6. Save results
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_image_list, f, indent=4)

    print("-" * 50)
    print(f"TOTAL IMAGES FETCHED: {len(final_image_list)}")
    print(f"EXPANDED MANIFEST SAVED TO: {output_json_path}")

if __name__ == "__main__":
    N_AVG = 14 
    
    TARGET_LIST = [
        "Trade Finance_20210518201634_1.00_page_33", "Trade Finance_20210406164948_1.00_page_29",
        "Trade Finance_20210602072658_1.00_page_7", "Trade Finance_20210208080309_1.00_page_12",
        "Trade Finance_20210410122700_1.00_page_17", "A09862_20210125165541_1.00_page_6",
        "Trade Finance_20210810165536_1.00_page_15", "Trade Finance_20210830144103_1.00_page_2",
        "Trade Finance_20210216165325_1.00_page_19", "Trade Finance_20210427073859_1.00_page_18",
        "Trade Finance_20210406174100_1.00_page_23", "Trade Finance_20210822180605_1.00_page_47",
        "Trade Finance_20210822174205_1.00_page_17", "Trade Finance_20210610140215_1.00_page_13",
        "Trade Finance_20210628172355_1.00_page_35", "Trade Finance_20210316080503_1.00_page_8",
        "Trade Finance_20210210160946_1.00_page_2", "Trade Finance_20210809082249_1.00_page_9",
        "Trade Finance_20210626155149_1.00_page_3", "Trade Finance_20210322193339_1.00_page_4",
        "Trade Finance_20210318074714_1.00_page_47", "Trade Finance_20210520090408_1.00_page_11",
        "Trade Finance_20210705160913_1.00_page_6", "Trade Finance_20210628171753_1.00_page_47",
        "Trade Finance_20210802164308_1.00_page_11", "Trade Finance_20210701163204_1.00_page_5",
        "Trade Finance_20210208145328_1.00_page_21", "A09893_20210128182637_1.00_page_10",
        "Trade Finance_20210208172125_1.00_page_2", "Trade Finance_20210617190002_1.00_page_6",
        "Trade Finance_20210606080110_1.00_page_4", "Trade Finance_20210208071246_1.00_page_5",
        "Trade Finance_20210527172618_1.00_page_2", "Trade Finance_20210410121612_1.00_page_6",
        "Trade Finance_20210901174621_1.00_page_4", "Trade Finance_20210608083123_1.00_page_34",
        "Trade Finance_20210608141637_1.00_page_2", "Trade Finance_20210825133302_1.00_page_7"
    ]
    
    IN_JSON = "/datadrive2/IDF_AL_MASRAF/balanced_diverse_pages_100.json"
    OUT_JSON = "targeted_sibling_expansion.json"

    collect_dynamic_siblings(IN_JSON, OUT_JSON, TARGET_LIST, n_avg=N_AVG)