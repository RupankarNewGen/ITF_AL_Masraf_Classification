'''this code is to pick random subfolders from a source
directory until we have approximately 500 images in total.'''



import os
import shutil
import random
from tqdm import tqdm

def copy_balanced_subfolders(source_root, destination_root, target_image_total=500):
    """
    Selects random subfolders from source_root until the total 
    image count is approximately target_image_total.
    """
    # 1. Setup paths
    if not os.path.exists(destination_root):
        os.makedirs(destination_root)
    
    # 2. Map out all subfolders and their image counts
    subfolder_data = [] # List of tuples: (subfolder_name, full_path, image_count)
    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    
    print(f"Scanning source directory: {source_root}")
    # Only look at the first level of directories
    subdirs = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(source_root, subdir)
        images = [f for f in os.listdir(subdir_path) if os.path.splitext(f)[1].lower() in valid_extensions]
        if images:
            subfolder_data.append((subdir, subdir_path, len(images)))

    # 3. Randomize and Select
    random.shuffle(subfolder_data)
    
    selected_subfolders = []
    current_total_images = 0
    
    for name, path, count in subfolder_data:
        # Check if adding this folder brings us closer to 500 or exceeds it significantly
        # We allow it to go slightly over to ensure we get a full folder
        selected_subfolders.append((name, path, count))
        current_total_images += count
        
        if current_total_images >= target_image_total:
            break

    print(f"Selected {len(selected_subfolders)} subfolders for a total of {current_total_images} images.")

    # 4. Copy the selected folders
    for name, path, count in tqdm(selected_subfolders, desc="Copying Folders"):
        dest_path = os.path.join(destination_root, name)
        try:
            # copytree copies the entire directory
            shutil.copytree(path, dest_path, dirs_exist_ok=True)
        except Exception as e:
            print(f"Error copying {name}: {e}")

    print("-" * 50)
    print(f"COMPLETED")
    print(f"Destination: {destination_root}")
    print(f"Final Image Count: {current_total_images}")
    print("-" * 50)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # This is the folder created in the previous step (containing many subfolders)
    SOURCE = "/datadrive2/IDF_AL_MASRAF/final_balanced_sample_images"
    
    # This is where the ~500 images (as full subfolders) will go
    DESTINATION = "/datadrive2/IDF_AL_MASRAF/500_images_from_final_balanced_sample"
    
    copy_balanced_subfolders(SOURCE, DESTINATION, target_image_total=500)