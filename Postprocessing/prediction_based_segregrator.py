import os
import json
from tqdm import tqdm

def organize_metadata_by_class(json_results_folder, manifest_json_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 1. Load the Manifest JSON
    print(f"Loading manifest from: {manifest_json_path}")
    try:
        with open(manifest_json_path, 'r', encoding='utf-8') as f:
            manifest_list = json.load(f)
    except Exception as e:
        print(f"Error loading manifest: {e}")
        return

    # 2. Index the Manifest: Map { 'base_name': '/abs/path/image.jpeg' }
    print("Indexing manifest paths...")
    image_path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in manifest_list}

    # 3. Storage for grouped paths
    # Format: { 'class_name': ['/path/1', '/path/2'] }
    class_groups = {}

    # 4. Get list of Result JSONs
    result_files = [f for f in os.listdir(json_results_folder) if f.endswith('.json')]
    
    missing_count = 0

    # 5. Process each Result
    for json_filename in tqdm(result_files, desc="Grouping Paths"):
        base_name = os.path.splitext(json_filename)[0]
        json_path = os.path.join(json_results_folder, json_filename)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                class_name = data.get("classification", "others").strip().replace(" ", "_")

            # 6. Lookup the absolute path and add to group
            if base_name in image_path_map:
                source_abs_path = image_path_map[base_name]
                
                if class_name not in class_groups:
                    class_groups[class_name] = []
                
                class_groups[class_name].append(source_abs_path)
            else:
                missing_count += 1

        except Exception as e:
            print(f"Error reading {json_filename}: {e}")

    # 7. Save each class group as its own JSON manifest
    print("\nSaving Class Manifests...")
    for class_label, paths in class_groups.items():
        output_file = os.path.join(output_directory, f"{class_label}.json")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(paths, out_f, indent=4)
        print(f" - Created {output_file} ({len(paths)} images)")

    print("\n" + "="*40)
    print("METADATA ORGANIZATION SUMMARY")
    print(f"Classes Identified: {len(class_groups)}")
    print(f"Missing in Manifest: {missing_count}")
    print("="*40)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    JSON_RESULTS_DIR = "/datadrive2/IDF_AL_MASRAF/LC_Drawing_Full_Result"
    MANIFEST_JSON = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/Import_LC_Drawing_manifest.json"
    # This folder will contain only small .json files
    OUTPUT_METADATA_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/lc_drawing_segregrated_full"

    organize_metadata_by_class(JSON_RESULTS_DIR, MANIFEST_JSON, OUTPUT_METADATA_DIR)