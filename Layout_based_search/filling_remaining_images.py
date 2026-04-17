import os
import shutil
from tqdm import tqdm

def get_leaf_folders(base_path):
    """Identifies folders that contain files directly."""
    leaf_folders = []
    for root, dirs, files in os.walk(base_path):
        if files:
            leaf_folders.append(os.path.relpath(root, base_path))
    return list(set(leaf_folders))

def create_supplemental_dataset(primary_dir, secondary_dir, output_dir, target_count=10):
    print(f"Counting gaps in {primary_dir}...")
    leaves = get_leaf_folders(primary_dir)
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

    for leaf in tqdm(leaves, desc="Finding Supplemental Images"):
        # Paths for counting (A) and sourcing (B)
        path_a = os.path.join(primary_dir, leaf)
        path_b = os.path.join(secondary_dir, leaf)
        dest_path = os.path.join(output_dir, leaf)

        # 1. Count how many images are currently in the Primary folder
        images_a = [f for f in os.listdir(path_a) if os.path.splitext(f)[1].lower() in valid_exts]
        current_count = len(images_a)

        # 2. If we have less than 10, find the supplemental images from Folder B
        if current_count < target_count:
            needed = target_count - current_count
            
            if os.path.exists(path_b):
                # Look at available images in Source B
                images_b = [f for f in os.listdir(path_b) if os.path.splitext(f)[1].lower() in valid_exts]
                
                # Exclude filenames already present in A to ensure we are adding "new" data
                unique_to_b = [img for img in images_b if img not in images_a]
                
                # Take only the amount needed to reach the total of 10
                supplemental_selection = unique_to_b[:needed]
                
                if supplemental_selection:
                    os.makedirs(dest_path, exist_ok=True)
                    for img in supplemental_selection:
                        shutil.copy2(os.path.join(path_b, img), os.path.join(dest_path, img))

    print(f"\nSupplemental images saved to: {output_dir}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Used for counting current images
    PRIMARY_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/al_masraf_2000_filtered"
    
    # The actual source of the new images
    SECONDARY_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/validated_similar_layouts_almasraf_2000"
    
    # Will only contain the "fill-in" images from Secondary
    OUTPUT_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/addtional_images_similar_to_2000"

    create_supplemental_dataset(PRIMARY_DIR, SECONDARY_DIR, OUTPUT_DIR, target_count=10)