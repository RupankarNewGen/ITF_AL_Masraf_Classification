import os
import json
import pandas as pd
import re

json_dir = "/home/lpt6964/Downloads/temp/updated_thresh"

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            json_path = os.path.join(directory, filename)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                result_dict = data.get('result', {})
                records = []
                
                for key, val in result_dict.items():
                    if not isinstance(val, dict):
                        continue
                        
                    # Extract page number from format like "input_tif_file__0.png"
                    page_match = re.search(r'__(\d+)\.\w+$', key, re.IGNORECASE)
                    if page_match:
                        page_no = f"page {int(page_match.group(1))}"
                    else:
                        page_no = key
                    
                    predicted_class = val.get("predicted_class", "")
                    confidence = val.get("confidence", "")
                    
                    records.append({
                        "page_no": page_no,
                        "predicted_class": predicted_class,
                        "confidence": confidence
                    })
                    
                if records:
                    df = pd.DataFrame(records)
                    
                    def get_page_int(p_str):
                        m = re.search(r'\d+', str(p_str))
                        return int(m.group()) if m else -1
                    
                    # Sort records logically by page number
                    df['_sort'] = df['page_no'].apply(get_page_int)
                    df = df.sort_values('_sort').drop('_sort', axis=1)
                    
                    excel_filename = filename.replace('.json', '.xlsx')
                    excel_path = os.path.join(directory, excel_filename)
                    
                    df.to_excel(excel_path, index=False)
                    print(f"Generated: {excel_filename}")
                else:
                    print(f"No valid records found for: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    process_directory(json_dir)
