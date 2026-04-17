import os
import json
import re

INPUT_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/New_Test_Data_jsons"
OUTPUT_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/New_test_data_jsons_splitted"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files in {INPUT_DIR}")
    
    processed_files = 0
    generated_jsons = 0
    
    for filename in files:
        base_name = filename.replace('.json', '')
        input_path = os.path.join(INPUT_DIR, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing {filename}: {e}")
                continue
                
        result_dict = data.get("result", {})
        if not result_dict:
            result_dict = {k: v for k, v in data.items() if isinstance(v, dict) and k != "result"}
            if not result_dict:
                continue
            
        # Create output directory for this workitem
        workitem_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(workitem_dir, exist_ok=True)
        
        for key, value in result_dict.items():
            # Extract number from 'input_tif_file__0.png'
            match = re.search(r'__(\d+)\.\w+$', key)
            if match:
                page_index = int(match.group(1))
            else:
                # If structure is different, fallback to 0 or extract digits
                digit_match = re.search(r'(\d+)', key)
                page_index = int(digit_match.group(1)) if digit_match else 0
                
            page_num = page_index + 1
            image_name = f"{base_name}_page_{page_num}.jpeg"
            
            # Format expected by phase 2
            single_json_content = {
                image_name: value
            }
            
            output_json_name = f"{base_name}_page_{page_num}.json"
            output_json_path = os.path.join(workitem_dir, output_json_name)
            
            with open(output_json_path, 'w', encoding='utf-8') as out_f:
                json.dump(single_json_content, out_f, indent=4)
                
            generated_jsons += 1
            
        processed_files += 1

    print(f"\n✅ Completed. Processed {processed_files} files.")
    print(f"Generated {generated_jsons} single-page JSON files in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
