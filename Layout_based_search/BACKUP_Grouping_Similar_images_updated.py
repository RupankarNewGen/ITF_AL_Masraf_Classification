"""
python ITF_Al_Masraf_Classification/Layout_based_search/Grouping_Similar_images_updated.py 
"""
import os
import shutil
import numpy as np
from imagededup.methods import PHash

def manual_hamming_distance(hash1, hash2):
    """Calculates the Hamming distance between two hex hashes."""
    # Ensure we are comparing equal length binary strings
    h1 = bin(int(hash1, 16))[2:].zfill(64)
    h2 = bin(int(hash2, 16))[2:].zfill(64)
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

def remove_duplicate_images_centroid(image_directory, unique_docs_dir, unique_duplicates_images_directory, threshold=15):
    phasher = PHash()
    print(f"Encoding images in {image_directory}...")
    encodings = phasher.encode_images(image_dir=image_directory)
    
    all_images = sorted(os.listdir(image_directory))
    visited = set()
    layout_id = 0

    # 1. Build a full adjacency map (Who is within threshold of whom?)
    # This respects your threshold properly.
    adj_map = {img: [] for img in all_images}
    print(f"Building similarity map (Threshold: {threshold})...")
    
    # Pre-calculate hashes to speed up the loop
    hash_list = [encodings.get(img) for img in all_images]

    for i in range(len(all_images)):
        if hash_list[i] is None: continue
        for j in range(i + 1, len(all_images)):
            if hash_list[j] is None: continue
            
            dist = manual_hamming_distance(hash_list[i], hash_list[j])
            if dist <= threshold:
                adj_map[all_images[i]].append(all_images[j])
                adj_map[all_images[j]].append(all_images[i])

    # 2. Greedy Centroid Clustering
    # We pick the image with the MOST unclaimed neighbors first to maximize group size.
    print("Clustering layouts...")
    while True:
        # Filter out already visited images from the map
        remaining_images = [img for img in all_images if img not in visited]
        if not remaining_images:
            break
            
        # Find the "Centroid" (the image with the most unclaimed neighbors)
        best_anchor = None
        best_neighbors = []
        
        for img in remaining_images:
            # Only count neighbors that haven't been visited
            unvisited_neighbors = [n for n in adj_map[img] if n not in visited]
            if best_anchor is None or len(unvisited_neighbors) > len(best_neighbors):
                best_anchor = img
                best_neighbors = unvisited_neighbors

        # If the best anchor has no neighbors, the rest are all "Unique/Others"
        if len(best_neighbors) == 0:
            break

        # Create the layout group
        layout_id += 1
        current_group = [best_anchor] + best_neighbors
        
        layout_path = os.path.join(unique_duplicates_images_directory, f"layout_{layout_id}")
        os.makedirs(layout_path, exist_ok=True)
        
        for img in current_group:
            src = os.path.join(image_directory, img)
            shutil.copy(src, os.path.join(layout_path, img))
            # Only copy unique docs once
            if img not in visited:
                shutil.copy(src, os.path.join(unique_docs_dir, img))
            visited.add(img)

    # 3. Final Step: Move truly unique images to "others"
    others = set(all_images) - visited
    others_path = os.path.join(unique_duplicates_images_directory, "others")
    os.makedirs(others_path, exist_ok=True)
    for img in others:
        src = os.path.join(image_directory, img)
        shutil.copy(src, os.path.join(others_path, img))
        shutil.copy(src, os.path.join(unique_docs_dir, img))

    print("\n" + "="*30)
    print(f"Total images: {len(all_images)}")
    print(f"Layouts found: {layout_id}")
    print(f"Unique images: {len(others)}")
    print("="*30)

if __name__ == "__main__":
    image_dir = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/FINAL_merged_classified_images/Packing_List"
    output_base = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/aishiki_FINAL_layouts_of_merged_classified_images/Packing_list2"
    
    image_dir = "/datadrive2/IDF_AL_MASRAF/Master_Repo/TEMP_DATA/5_Docs_New_test_Organized_by_Classification/CI"
    output_base = "/datadrive2/IDF_AL_MASRAF/Master_Repo/TEMP_DATA/5_Docs_LAYOUT_WISE_New_test_Organized_by_Classification/CI"

    image_dir = "/datadrive2/IDF_AL_MASRAF/Master_Repo/TEMP_DATA/5_Docs_New_test_Organized_by_Classification/BOE"
    output_base = "/datadrive2/IDF_AL_MASRAF/Master_Repo/TEMP_DATA/5_Docs_LAYOUT_WISE_New_test_Organized_by_Classification/BOE"
    
    unique_dup_dir = os.path.join(output_base, "duplicates_removal_res")
    unique_docs = os.path.join(output_base, "unique_docs")
    
    os.makedirs(unique_dup_dir, exist_ok=True)
    os.makedirs(unique_docs, exist_ok=True)
    
    # You can now easily set the threshold here (e.g., 18 for more inclusive groups)
    remove_duplicate_images_centroid(image_dir, unique_docs, unique_dup_dir, threshold=15)