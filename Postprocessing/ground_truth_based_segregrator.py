import os
import pandas as pd
import shutil
from tqdm import tqdm

def organize_by_csv(csv_path, source_root, output_root):
    # 1. Setup output directory
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output root: {output_root}")

    # 2. Load the CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} entries.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Index the Source Folder (Recursive Search)
    # We create a map of { 'filename_with_ext': 'full_absolute_path' }
    print("Indexing source images... (Scanning subfolders)")
    image_map = {}
    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    
    for root, _, files in os.walk(source_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_extensions:
                # We map the exact filename from disk to its full path
                image_map[f] = os.path.join(root, f)

    print(f"Indexed {len(image_map)} images on disk.")

    # 4. Process CSV and Copy
    copied_count = 0
    missing_count = 0

    # Iterate through CSV rows
    # Headers: Folder Name, images_name, Ground truth, Others
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing by Ground Truth"):
        img_filename = str(row['images_name']).strip()
        ground_truth = str(row['Ground truth']).strip()

        # Handle potential empty or nan ground truth
        if not ground_truth or ground_truth.lower() == 'nan':
            ground_truth = "unclassified"

        # Create the Ground Truth subfolder
        dest_class_folder = os.path.join(output_root, ground_truth)
        os.makedirs(dest_class_folder, exist_ok=True)

        # Check if the image exists in our indexed map
        if img_filename in image_map:
            src_path = image_map[img_filename]
            dest_path = os.path.join(dest_class_folder, img_filename)
            
            try:
                shutil.copy2(src_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Failed to copy {img_filename}: {e}")
        else:
            missing_count += 1

    print("\n" + "="*40)
    print(f"ORGANIZATION SUMMARY")
    print(f"Images Successfully Organized: {copied_count}")
    print(f"Images Not Found on Disk:     {missing_count}")
    print(f"Results stored in: {output_root}")
    print("="*40)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    CSV_FILE = "/datadrive2/IDF_AL_MASRAF/test_images/set2_first_500_images_ground_truth.csv"
    
    # The folder containing the many subfolders (TF19215..., TF20007..., etc.)
    SOURCE_DIR = "/datadrive2/IDF_AL_MASRAF/500_images_from_second_set_sample"
    
    # Where you want the Ground Truth folders created
    OUTPUT_DIR = "/datadrive2/IDF_AL_MASRAF/Al_MASRAF_seocond_set_first_500_categorized_data"

    organize_by_csv(CSV_FILE, SOURCE_DIR, OUTPUT_DIR)