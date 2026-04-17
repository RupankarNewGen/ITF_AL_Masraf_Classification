import os
import json

def subtract_json_paths(superset_json_path, subset_json_path, output_json_path):
    # ---------------------------------------------------------
    # STEP 1: Load both JSON files
    # ---------------------------------------------------------
    print(f"Loading superset from: {os.path.basename(superset_json_path)}...")
    try:
        with open(superset_json_path, 'r', encoding='utf-8') as f:
            superset_paths = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load Superset JSON: {e}")
        return

    print(f"Loading subset from: {os.path.basename(subset_json_path)}...")
    try:
        with open(subset_json_path, 'r', encoding='utf-8') as f:
            subset_paths = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load Subset JSON: {e}")
        return

    # ---------------------------------------------------------
    # STEP 2: Convert subset to a set for O(1) fast lookups
    # ---------------------------------------------------------
    subset_set = set(subset_paths)

    # ---------------------------------------------------------
    # STEP 3: Subtract the subset from the superset
    # ---------------------------------------------------------
    print("Subtracting subset from superset...")
    
    # We use a list comprehension here so the remaining paths 
    # keep the exact same order they had in the original superset.
    remaining_paths = [path for path in superset_paths if path not in subset_set]

    # ---------------------------------------------------------
    # STEP 4: Save the resulting list to a new JSON
    # ---------------------------------------------------------
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(remaining_paths, f, indent=4)
        
    print("-" * 50)
    print("=== SUBTRACTION SUMMARY ===")
    print(f"Original Superset Count: {len(superset_paths)}")
    print(f"Subset Count Removed:    {len(subset_paths)}")
    print(f"Remaining Paths Saved:   {len(remaining_paths)}")
    print(f"Saved to: {output_json_path}")
    print("-" * 50)


if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    
    # 1. The massive list of all paths
    SUPERSET_JSON = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/Import_LC_Drawing_manifest.json"
    
    # 2. The list of paths you want to remove (from your previous script)
    SUBSET_JSON = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/lcdrawing_images_already_used_in_traning.json"
    
    # 3. Where you want the leftovers saved
    OUTPUT_JSON = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/lc_drawing_remaining_images"
    
    subtract_json_paths(
        superset_json_path=SUPERSET_JSON,
        subset_json_path=SUBSET_JSON,
        output_json_path=OUTPUT_JSON
    )