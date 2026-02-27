import os
import shutil
from tqdm import tqdm

def copy_remaining_subfolders(primary_source, reference_subset, output_folder):
    """
    Identifies folders in primary_source that are NOT in reference_subset
    and copies them to the output_folder.
    """
    # 1. Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # 2. Get list of subfolder names in both locations
    # We only care about directories, ignoring any loose files
    primary_dirs = {d for d in os.listdir(primary_source) 
                    if os.path.isdir(os.path.join(primary_source, d))}
    
    reference_dirs = {d for d in os.listdir(reference_subset) 
                      if os.path.isdir(os.path.join(reference_subset, d))}

    # 3. Calculate the difference (A - B)
    remaining_dirs = sorted(list(primary_dirs - reference_dirs))

    print(f"Total folders in Primary: {len(primary_dirs)}")
    print(f"Total folders in Reference: {len(reference_dirs)}")
    print(f"Remaining to copy: {len(remaining_dirs)}")

    if not remaining_dirs:
        print("No remaining folders found to copy.")
        return

    # 4. Copy the folders
    for folder_name in tqdm(remaining_dirs, desc="Copying Remaining Folders"):
        src_path = os.path.join(primary_source, folder_name)
        dest_path = os.path.join(output_folder, folder_name)

        try:
            # copytree copies the entire directory structure
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        except Exception as e:
            print(f"Error copying {folder_name}: {e}")

    print("-" * 50)
    print(f"SUCCESS: {len(remaining_dirs)} folders copied to {output_folder}")
    print("-" * 50)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # The 'Main Folder A' (containing a, b, c, d, e) # bada folder with all the folders we want to filter from
    FOLDER_A = "/datadrive2/IDF_AL_MASRAF/final_balanced_sample_images_1000"
    
    # The 'Reference Folder' (containing c, d) # chota folder with only the subset of folders we want to exclude from copying
    FOLDER_REF = "/datadrive2/IDF_AL_MASRAF/500_images_from_second_set_sample"
    
    # The 'Destination' (where a, b, e will be saved)
    OUTPUT = "/datadrive2/IDF_AL_MASRAF/500_remaining_images_from_second_set_sample"

    copy_remaining_subfolders(FOLDER_A, FOLDER_REF, OUTPUT)