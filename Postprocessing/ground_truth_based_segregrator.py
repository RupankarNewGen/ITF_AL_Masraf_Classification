import os
import pandas as pd
import shutil
from tqdm import tqdm

def organize_by_csv(csv_path, source_root, output_root, exclude_root=None):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 1. Index Exclusions (By filename without extension)
    exclude_set = set()
    if exclude_root and os.path.exists(exclude_root):
        print(f"Indexing excluded images from: {exclude_root}")
        for root, _, files in os.walk(exclude_root):
            for f in files:
                name_no_ext = os.path.splitext(f)[0].strip()
                exclude_set.add(name_no_ext)
        print(f"Found {len(exclude_set)} entries in exclusion.")

    # 2. Load the CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} entries.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Index the Source Folder (Mapping Base Name -> Full Path)
    print("Indexing source images... (Removing extensions for matching)")
    image_map = {}
    valid_exts = {'.jpeg', '.jpg', '.png', '.tiff', '.bmp'}
    
    for root, _, files in os.walk(source_root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                # 'Trade Finance_..._page_1.jpeg' -> 'Trade Finance_..._page_1'
                name_no_ext = os.path.splitext(f)[0].strip()
                image_map[name_no_ext] = os.path.join(root, f)
            
    print(f"Indexed {len(image_map)} images on disk.")

    # 4. Process and Match
    copied_count = 0
    missing_count = 0
    skipped_excluded = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
        # exact string from CSV (no extension)
        csv_id = str(row['image name']).strip()
        ground_truth = str(row['Ground Truth']).strip()

        # Check exclusion set
        if csv_id in exclude_set:
            skipped_excluded += 1
            continue

        if not ground_truth or ground_truth.lower() == 'nan':
            ground_truth = "Other"
        
        dest_folder = os.path.join(output_root, ground_truth)
        os.makedirs(dest_folder, exist_ok=True)

        # SIMPLE MATCHING: Direct key lookup in the image_map
        if csv_id in image_map:
            src_path = image_map[csv_id]
            dest_path = os.path.join(dest_folder, os.path.basename(src_path))
            
            try:
                shutil.copy2(src_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Failed to copy {csv_id}: {e}")
        else:
            missing_count += 1

    print("\n" + "="*40)
    print(f"ORGANIZATION SUMMARY")
    print(f"Successfully Organized: {copied_count}")
    print(f"Skipped (Excluded):    {skipped_excluded}")
    print(f"Not Found:             {missing_count}")
    print("="*40)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    CSV_FILE = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Prediction_CSV/second_set_1000_images_results(in).csv"
    SOURCE_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_input_Data/500_remaining_images_from_second_set_sample"
    OUTPUT_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/Al_Masraf_second_set_remaining_500_categorized_data"
    EXCLUDE_DIR = "/datadrive2/IDF_AL_MASRAF/Al_MASRAF_seocond_set_first_500_categorized_data"

    organize_by_csv(CSV_FILE, SOURCE_DIR, OUTPUT_DIR, exclude_root=EXCLUDE_DIR)