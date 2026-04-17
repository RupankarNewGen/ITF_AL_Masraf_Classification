import os
import json
import shutil
import numpy as np
from imagededup.methods import PHash, AHash, CNN

def manual_hamming_distance(hash1, hash2):
    """Calculates the Hamming distance between two hex/numpy hashes."""
    return sum(c1 != c2 for c1, c2 in zip(bin(int(hash1, 16)), bin(int(hash2, 16))))

def load_exclusion_list(exclusion_sources):
    """
    Load exclusion list from multiple JSON files or recursively from folders.
    """
    exclusion_set = set()
    valid_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    
    if exclusion_sources is None:
        return exclusion_set
    
    if isinstance(exclusion_sources, str):
        exclusion_sources = [exclusion_sources]
    
    for source in exclusion_sources:
        if not os.path.exists(source):
            print(f"⚠️  Exclusion source not found: {source}")
            continue
        
        # --- NEW LOGIC: Check if the source is a directory ---
        if os.path.isdir(source):
            print(f"📂 Scanning exclusion folder: {source}")
            try:
                for root, dirs, files in os.walk(source):
                    for file in files:
                        if os.path.splitext(file)[1].lower() in valid_exts:
                            exclusion_set.add(file) # 'file' is already the basename
                print(f"✓ Loaded items from folder: {os.path.basename(source)}")
            except Exception as e:
                print(f"✗ Error scanning folder {source}: {e}")
                
        # --- ORIGINAL LOGIC: Check if the source is a file (JSON) ---
        elif os.path.isfile(source):
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    exclusion_list = json.load(f)
                
                if isinstance(exclusion_list, list):
                    for path in exclusion_list:
                        exclusion_set.add(os.path.basename(path))
                elif isinstance(exclusion_list, dict):
                    for key in exclusion_list.keys():
                        exclusion_set.add(os.path.basename(key))
                
                print(f"✓ Loaded items from JSON: {os.path.basename(source)}")
            except Exception as e:
                print(f"✗ Error loading JSON {source}: {e}")
            
    print(f"Total unique items in combined exclusion list: {len(exclusion_set)}")
    return exclusion_set


def find_and_save_matches(input_path, search_source, main_output_folder, 
                         method='phash', exclusion_source=None):
    
    exclusion_set = load_exclusion_list(exclusion_source)
    
    if method.lower() == 'phash':
        hasher = PHash()
    elif method.lower() == 'ahash':
        hasher = AHash()
    elif method.lower() == 'cnn':                                                                 
        hasher = CNN()
    else:
        raise ValueError("Invalid method. Choose 'phash', 'ahash', or 'cnn'.")

    print(f"\n{'='*80}")
    print(f"SIMILAR LAYOUT FINDER - {method.upper()} Hashing (Auto-Threshold 10-18)")
    print(f"{'='*80}\n")
    
    folder_encodings = {}
    search_paths = []

    if search_source.endswith('.json'):
        print(f"📂 Loading search paths from manifest: {search_source}")
        with open(search_source, 'r', encoding='utf-8') as f:
            search_paths = json.load(f)
        
        print(f"🔄 Encoding {len(search_paths)} images from manifest...")
        for p in search_paths:
            try:
                encoding = hasher.encode_image(p)
                folder_encodings[os.path.basename(p)] = encoding
            except Exception as e:
                print(f"⚠️  Skipping {p} due to error: {e}")
    else:
        print(f"📂 Scanning search folder: {search_source}")
        folder_encodings = hasher.encode_images(image_dir=search_source)
        search_paths = [os.path.join(search_source, f) for f in os.listdir(search_source)]
    
    full_path_map = {os.path.basename(p): p for p in search_paths}
    valid_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

    # --- INPUT LOGIC: Determine Processing Mode ---
    targets = [] 

    if os.path.isfile(input_path):
        # Case 1: Single image
        targets.append((input_path, "")) 
    else:
        items = os.listdir(input_path)
        subdirs = [d for d in items if os.path.isdir(os.path.join(input_path, d))]
        images = [f for f in items if os.path.splitext(f)[1].lower() in valid_exts]

        if subdirs:
            # Case 3: Folder of subfolders
            for sd in subdirs:
                sd_path = os.path.join(input_path, sd)
                sd_images = [f for f in os.listdir(sd_path) if os.path.splitext(f)[1].lower() in valid_exts]
                if sd_images:
                    targets.append((os.path.join(sd_path, sorted(sd_images)[0]), sd))
        else:
            # Case 2: Folder of images
            for img in images:
                targets.append((os.path.join(input_path, img), os.path.splitext(img)[0]))

    print(f"🎯 Found {len(targets)} target images to process\n")
    
    stats = {'processed': 0, 'excluded': 0, 'matches_found': {}, 'excluded_matches': {}}

    for target_image_path, subfolder_name in targets:
        target_filename = os.path.basename(target_image_path)
        output_folder = os.path.join(main_output_folder, subfolder_name)
        
        print(f"Processing: {target_filename} (Subfolder: {subfolder_name if subfolder_name else 'Root'})...", end=" ", flush=True)
        target_encoding = hasher.encode_image(target_image_path)

        if method.lower() == 'cnn':
            if len(target_encoding.shape) > 1:
                target_encoding = target_encoding.flatten()

        current_map = folder_encodings.copy()
        current_map[target_filename] = target_encoding

        # --- AUTO-THRESHOLD LOGIC STARTS HERE ---
        # CNN doesn't use hamming distance, so it just runs once. Phash/Ahash uses the 10-18 range.
        thresholds_to_test = [0.90] if method.lower() == 'cnn' else [10, 12, 14, 16, 18]
        
        final_included_matches = []
        final_excluded_matches = []
        final_threshold = None

        for current_threshold in thresholds_to_test:
            if method.lower() == 'cnn':
                search_params = {'min_similarity_threshold': current_threshold}
            else:
                search_params = {'max_distance_threshold': current_threshold}

            all_duplicates = hasher.find_duplicates(encoding_map=current_map, **search_params)
            matches = all_duplicates.get(target_filename, [])
            
            temp_excluded = []
            temp_included = []
            
            for filename in matches:
                if filename == target_filename:
                    continue
                if filename in exclusion_set:
                    temp_excluded.append(filename)
                else:
                    temp_included.append(filename)
            
            # Update our final lists with the results of this threshold pass
            final_included_matches = temp_included
            final_excluded_matches = temp_excluded
            final_threshold = current_threshold
            
            # Check if we hit our quota of 10!
            if len(final_included_matches) >= 10:
                break # We have enough images, break out of the threshold loop early!
                
        # --- AUTO-THRESHOLD LOGIC ENDS HERE ---

        stats['processed'] += 1
        stats['matches_found'][target_filename] = len(final_included_matches)
        if final_excluded_matches:
            stats['excluded_matches'][target_filename] = len(final_excluded_matches)
        
        # Finally, save the results based on the best threshold found
        if final_included_matches or True:
            os.makedirs(output_folder, exist_ok=True)
            for filename in final_included_matches:
                source_path = full_path_map.get(filename)
                if source_path and os.path.exists(source_path):
                    shutil.copy(source_path, os.path.join(output_folder, filename))
            
            if final_excluded_matches:
                report_path = os.path.join(output_folder, "EXCLUDED_MATCHES.txt")
                with open(report_path, 'a') as f:
                    f.write(f"\nTarget: {target_filename}\nExcluded {len(final_excluded_matches)} images at threshold {final_threshold}:\n")
                    for filename in sorted(final_excluded_matches):
                        f.write(f"  - {filename}\n")
            
            print(f"✓ {len(final_included_matches)} matches (Stopped at Threshold: {final_threshold})")
        else:
            print(f"⊘ All matches excluded")

    # Summary Report
    print(f"\n{'='*80}\nSIMILAR LAYOUT SUMMARY\n{'='*80}\n")
    for img, count in sorted(stats['matches_found'].items()):
        print(f"  {img:50} : {count:>4} matches")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    
    input_folder_path = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/aishiki_AlmasRaf_test_pdfs/test_set2_copy/PL" # Can be a single image, a folder of images, or a folder of subfolders with images
    
    search_source = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/lc_drawing_segregrated_full/Packing_List.json" # Can be a folder of images or a JSON manifest of image paths
    main_output_path = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/test_set2_similar_images/PL" # Output folder where results will be saved
    
    # You can now put a mix of JSON files AND Folders here:
    exclusion_source = [
        "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/2000_images_set_march_5_2026.json",
        "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/remaining_filing_images_from_8000",
        "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/remianing_images_to_reach_10",
        "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/addtional_images_similar_to_2000"
    ]
    
    find_and_save_matches(
        input_folder_path, 
        search_source, 
        main_output_path, 
        method='phash',
        exclusion_source=exclusion_source
    )