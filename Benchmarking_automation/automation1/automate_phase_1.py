import json
import os
import subprocess
import sys

# Configuration
FINAL_MATCHED_DIRECTORIES_JSON = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/FINAL_MATCHED_DIRECTORIES.json"
GET_IMAGES_SCRIPT = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Preprocessing/get_images_from_annoation_folders.py"

def main():
    if not os.path.exists(FINAL_MATCHED_DIRECTORIES_JSON):
        print(f"Error: Could not find {FINAL_MATCHED_DIRECTORIES_JSON}")
        return

    with open(FINAL_MATCHED_DIRECTORIES_JSON, 'r', encoding='utf-8') as f:
        directories = json.load(f)

    if not isinstance(directories, list):
        print("Error: JSON must contain a list of directories.")
        return

    print(f"Found {len(directories)} directories to process.")

    env = os.environ.copy()
    
    for idx, directory in enumerate(directories, 1):
        print(f"\n[{idx}/{len(directories)}] Processing directory: {directory}")
        try:
            # We use the qwen_env environment as requested by the user.
            subprocess.run([
                "/datadrive2/IDF_AL_MASRAF/qwen_env/bin/python",
                GET_IMAGES_SCRIPT,
                "--directory", directory
            ], env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {directory}: {e}")
            
    print("\n✅ Phase 1 completed successfully.")

if __name__ == "__main__":
    main()
