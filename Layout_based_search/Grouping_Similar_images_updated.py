"""
Batch process classification folders for layout-wise duplicate removal and grouping.
python ITF_Al_Masraf_Classification/Layout_based_search/Grouping_Similar_images_updated.py 
"""
import os
import shutil
import json
from pathlib import Path
from imagededup.methods import PHash
from collections import defaultdict

def manual_hamming_distance(hash1, hash2):
    """Calculates the Hamming distance between two hex hashes."""
    # Ensure we are comparing equal length binary strings
    h1 = bin(int(hash1, 16))[2:].zfill(64)
    h2 = bin(int(hash2, 16))[2:].zfill(64)
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

def remove_duplicate_images_centroid(image_directory, unique_docs_dir, unique_duplicates_images_directory, threshold=15, save_unique_dir=True):
    """
    Remove duplicate images and group by layout using centroid clustering.
    
    Args:
        image_directory: Input folder with images
        unique_docs_dir: Output folder for unique documents
        unique_duplicates_images_directory: Output folder for layout groups
        threshold: Hamming distance threshold for similarity
        save_unique_dir: Whether to save unique_docs folder (can be disabled to save space)
    
    Returns:
        Dictionary with clustering statistics
    """
    
    phasher = PHash()
    print(f"  📸 Encoding images in {os.path.basename(image_directory)}...")
    encodings = phasher.encode_images(image_dir=image_directory)
    
    all_images = sorted(os.listdir(image_directory))
    visited = set()
    layout_id = 0
    
    stats = {
        'total_images': len(all_images),
        'layouts_found': 0,
        'unique_images': 0,
        'layout_groups': {}
    }

    # 1. Build a full adjacency map (Who is within threshold of whom?)
    adj_map = {img: [] for img in all_images}
    print(f"  🔗 Building similarity map (Threshold: {threshold})...")
    
    # Pre-calculate hashes to speed up the loop
    hash_list = [encodings.get(img) for img in all_images]

    for i in range(len(all_images)):
        if hash_list[i] is None: 
            continue
        for j in range(i + 1, len(all_images)):
            if hash_list[j] is None: 
                continue
            
            dist = manual_hamming_distance(hash_list[i], hash_list[j])
            if dist <= threshold:
                adj_map[all_images[i]].append(all_images[j])
                adj_map[all_images[j]].append(all_images[i])

    # 2. Greedy Centroid Clustering
    print("  🎯 Clustering layouts...")
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
        
        # Store statistics
        stats['layout_groups'][f"layout_{layout_id}"] = len(current_group)
        
        for img in current_group:
            src = os.path.join(image_directory, img)
            shutil.copy(src, os.path.join(layout_path, img))
            # Only copy unique docs if enabled
            if save_unique_dir and unique_docs_dir and img not in visited:
                shutil.copy(src, os.path.join(unique_docs_dir, img))
            visited.add(img)

    # 3. Final Step: Move truly unique images to "others"
    others = set(all_images) - visited
    if others:
        others_path = os.path.join(unique_duplicates_images_directory, "others")
        os.makedirs(others_path, exist_ok=True)
        
        for img in others:
            src = os.path.join(image_directory, img)
            shutil.copy(src, os.path.join(others_path, img))
            if save_unique_dir and unique_docs_dir:
                shutil.copy(src, os.path.join(unique_docs_dir, img))

    stats['layouts_found'] = layout_id
    stats['unique_images'] = len(others)
    
    return stats


def process_single_folder(input_folder, output_base_folder, threshold=15, 
                         save_unique_dir=True):
    """
    Process a single classification folder.
    
    Args:
        input_folder: Input folder with images
        output_base_folder: Base output folder (will create subfolder structure)
        threshold: Hamming distance threshold
        save_unique_dir: Whether to save unique_docs folder
    
    Returns:
        Tuple of (folder_name, statistics)
    """
    
    folder_name = os.path.basename(input_folder)
    
    # Create output structure
    output_folder = os.path.join(output_base_folder, folder_name)
    unique_dup_dir = os.path.join(output_folder, "duplicates_removal_res")
    
    # Only create unique_docs_dir path if save_unique_dir is True
    if save_unique_dir:
        unique_docs_dir = os.path.join(output_folder, "unique_docs")
        os.makedirs(unique_docs_dir, exist_ok=True)
    else:
        unique_docs_dir = None
    
    os.makedirs(unique_dup_dir, exist_ok=True)
    
    # Process the folder
    stats = remove_duplicate_images_centroid(
        input_folder, 
        unique_docs_dir, 
        unique_dup_dir, 
        threshold=threshold,
        save_unique_dir=save_unique_dir
    )
    
    return folder_name, stats


def get_level1_subfolders(base_folder):
    """
    Get all level-1 subfolders from a base folder.
    
    Args:
        base_folder: Base folder path
    
    Returns:
        Sorted list of subfolder names
    """
    
    subfolders = []
    
    if not os.path.isdir(base_folder):
        print(f"✗ Base folder not found: {base_folder}")
        return subfolders
    
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path):
            subfolders.append(item)
    
    return sorted(subfolders)


def load_folder_list_from_json(json_path):
    """
    Load folder list from JSON file.
    
    Args:
        json_path: Path to JSON file containing list of folder names
    
    Returns:
        List of folder names
    """
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            folders = json.load(f)
        
        if not isinstance(folders, list):
            print(f"✗ JSON must contain a list of folder names")
            return []
        
        return folders
    
    except Exception as e:
        print(f"✗ Error loading JSON: {e}")
        return []


def process_multiple_folders(base_input_folder, base_output_folder, 
                            folder_names=None, threshold=15, 
                            save_unique_dir=True):
    """
    Process multiple level-1 classification folders.
    
    Args:
        base_input_folder: Base folder containing classification subfolders
        base_output_folder: Base output folder
        folder_names: List of specific subfolder names to process
                     If None, processes all subfolders
        threshold: Hamming distance threshold
        save_unique_dir: Whether to save unique_docs folder
    
    Returns:
        Dictionary with statistics for all processed folders
    """
    
    print(f"\n{'='*80}")
    print(f"LAYOUT-WISE DUPLICATE REMOVAL & GROUPING (BATCH MODE)")
    print(f"{'='*80}\n")
    
    print(f"📂 Base Input Folder: {base_input_folder}")
    print(f"📂 Base Output Folder: {base_output_folder}")
    print(f"⚙️  Threshold: {threshold}")
    print(f"💾 Save Unique Dir: {save_unique_dir}\n")
    
    # Validate input folder
    if not os.path.isdir(base_input_folder):
        print(f"✗ Input folder not found: {base_input_folder}")
        return {}
    
    # Get list of folders to process
    if folder_names is None:
        # Process all subfolders
        folder_names = get_level1_subfolders(base_input_folder)
        print(f"🔄 Mode: Process ALL level-1 subfolders")
    else:
        print(f"🔄 Mode: Process SPECIFIC subfolders")
    
    if not folder_names:
        print(f"✗ No folders found to process")
        return {}
    
    print(f"📊 Folders to process: {len(folder_names)}")
    for fname in folder_names:
        print(f"   ✓ {fname}")
    print()
    
    all_stats = {}
    successful = 0
    failed = 0
    
    # Process each folder
    for idx, folder_name in enumerate(folder_names, 1):
        input_folder = os.path.join(base_input_folder, folder_name)
        
        # Validate folder exists
        if not os.path.isdir(input_folder):
            print(f"\n[{idx}/{len(folder_names)}] ⚠️  Folder not found: {folder_name}")
            failed += 1
            continue
        
        # Check if folder has images
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif', '.webp'}
        images = [f for f in os.listdir(input_folder) 
                 if os.path.splitext(f)[1].lower() in valid_extensions]
        
        if not images:
            print(f"\n[{idx}/{len(folder_names)}] ⚠️  No images found in: {folder_name}")
            failed += 1
            continue
        
        print(f"\n[{idx}/{len(folder_names)}] 📁 {folder_name} ({len(images)} images)")
        
        try:
            folder_result, stats = process_single_folder(
                input_folder, 
                base_output_folder, 
                threshold=threshold,
                save_unique_dir=save_unique_dir
            )
            
            all_stats[folder_result] = stats
            successful += 1
            
            # Print folder statistics
            print(f"  ✓ Total images: {stats['total_images']}")
            print(f"  ✓ Layouts found: {stats['layouts_found']}")
            print(f"  ✓ Unique images: {stats['unique_images']}")
        
        except Exception as e:
            print(f"  ✗ Error processing {folder_name}: {e}")
            failed += 1
    
    # Print summary report
    print_summary_report(all_stats, base_output_folder, successful, failed)
    
    # Save statistics to file
    save_statistics(all_stats, base_output_folder)
    
    return all_stats


def print_summary_report(all_stats, output_folder, successful, failed):
    """
    Print summary report of all processed folders.
    
    Args:
        all_stats: Dictionary with statistics for all folders
        output_folder: Output base folder
        successful: Number of successfully processed folders
        failed: Number of failed folders
    """
    
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}\n")
    
    if not all_stats:
        print(f"No folders were processed.")
        return
    
    total_images = 0
    total_layouts = 0
    total_unique = 0
    
    print(f"{'Classification':<25} {'Images':<12} {'Layouts':<12} {'Unique':<12}")
    print(f"{'-'*61}")
    
    for folder_name in sorted(all_stats.keys()):
        stats = all_stats[folder_name]
        total_images += stats['total_images']
        total_layouts += stats['layouts_found']
        total_unique += stats['unique_images']
        
        print(f"{folder_name:<25} {stats['total_images']:<12} "
              f"{stats['layouts_found']:<12} {stats['unique_images']:<12}")
    
    print(f"{'-'*61}")
    print(f"{'TOTAL':<25} {total_images:<12} {total_layouts:<12} {total_unique:<12}")
    
    print(f"\n{'='*80}")
    print(f"Output Directory Structure:")
    print(f"  {os.path.basename(output_folder)}/")
    
    for folder_name in sorted(all_stats.keys()):
        print(f"  ├─ {folder_name}/")
        print(f"  │  ├─ unique_docs/ (optional)")
        print(f"  │  └─ duplicates_removal_res/")
        print(f"  │     ├─ layout_1/")
        print(f"  │     ├─ layout_2/")
        print(f"  │     └─ others/")
    
    print(f"\n{'='*80}\n")


def save_statistics(all_stats, output_folder):
    """
    Save processing statistics to JSON file.
    
    Args:
        all_stats: Dictionary with statistics
        output_folder: Output folder
    """
    
    os.makedirs(output_folder, exist_ok=True)
    stats_file = os.path.join(output_folder, "processing_statistics.json")
    
    # Format statistics for JSON
    stats_for_json = {}
    for folder_name, stats in all_stats.items():
        stats_for_json[folder_name] = {
            'total_images': stats['total_images'],
            'layouts_found': stats['layouts_found'],
            'unique_images': stats['unique_images'],
            'layout_groups': stats.get('layout_groups', {})
        }
    
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_for_json, f, indent=2)
        
        print(f"✓ Statistics saved to: {stats_file}")
    
    except Exception as e:
        print(f"✗ Error saving statistics: {e}")


if __name__ == "__main__":
    
    # ========== CONFIGURATION ==========
    
    # Base input folder containing classification subfolders (level-1)
    base_input_folder = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/BOE"
    
    # Base output folder where classification subfolders will be created
    base_output_folder = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/BOE_Segregrated"
    
    # Hamming distance threshold for similarity
    threshold = 15
    
    # Whether to save unique_docs folder (set to False to save disk space)
    save_unique_docs = False
    
    # ========== MODE 1: Process ALL subfolders ==========
    process_multiple_folders(
        base_input_folder=base_input_folder,
        base_output_folder=base_output_folder,
        folder_names=None,  # None = process all subfolders
        threshold=threshold,
        save_unique_dir=save_unique_docs
    )
    
    # ========== MODE 2: Process SPECIFIC subfolders (uncomment to use) ==========
    # process_multiple_folders(
    #     base_input_folder=base_input_folder,
    #     base_output_folder=base_output_folder,
    #     folder_names=['CI', 'BOE', 'BOL'],  # Only these folders
    #     threshold=threshold,
    #     save_unique_dir=save_unique_docs
    # )
    
    # ========== MODE 3: Load folder names from JSON (uncomment to use) ==========
    # folder_list_json = "/path/to/folder_list.json"
    # folder_names = load_folder_list_from_json(folder_list_json)
    # process_multiple_folders(
    #     base_input_folder=base_input_folder,
    #     base_output_folder=base_output_folder,
    #     folder_names=folder_names,
    #     threshold=threshold,
    #     save_unique_dir=save_unique_docs
    # )