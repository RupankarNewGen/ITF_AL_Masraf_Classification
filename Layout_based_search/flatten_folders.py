import os
import shutil

def flatten_folders(base_path):
    print(f"Scanning directory: {base_path}\n")
    
    # 1. Get all the Level 1 folders (e.g., L1, L2)
    level1_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    for l1_name in level1_folders:
        l1_path = os.path.join(base_path, l1_name)
        
        # 2. Get all the Level 2 folders inside the current Level 1 folder (e.g., p1, p2)
        level2_folders = [f for f in os.listdir(l1_path) if os.path.isdir(os.path.join(l1_path, f))]
        
        for l2_name in level2_folders:
            old_l2_path = os.path.join(l1_path, l2_name)
            
            # 3. Create the new merged name (e.g., L1_p1)
            new_folder_name = f"{l1_name}_{l2_name}"
            new_folder_path = os.path.join(base_path, new_folder_name)
            
            # 4. Move the folder to the new location
            print(f"Moving: {l1_name}/{l2_name}  --->  {new_folder_name}")
            shutil.move(old_l2_path, new_folder_path)
            
        # 5. Clean up: Try to remove the old L1 folder if it's empty
        try:
            if not os.listdir(l1_path): # Checks if folder is empty
                os.rmdir(l1_path)
                print(f"Removed empty folder: {l1_name}")
            else:
                print(f"⚠️ Could not remove {l1_name} because it still contains files.")
        except Exception as e:
            print(f"Error removing {l1_name}: {e}")

    print("\n✅ Folder flattening complete!")

if __name__ == "__main__":
    # Put the path to your main "layouts" folder here
    INPUT_FOLDER = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/al_masraf_2000_filtered/CS"
    
    flatten_folders(INPUT_FOLDER)