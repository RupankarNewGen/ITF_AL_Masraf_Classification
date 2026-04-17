import os
import json
import argparse

def gen_custom_data(root_path, ocr_file):
    custom_path = os.path.join(root_path, "custom_data")
    if not os.path.exists(custom_path):
        os.makedirs(custom_path, exist_ok=True)
        
    all_words_path = os.path.join(custom_path, "all_words")
    if not os.path.exists(all_words_path):
        os.makedirs(all_words_path, exist_ok=True)

    import ast

    # Filtering for TXT or JSON files containing Abbyy OCR data
    filtered_filenames = [
        filename for filename in os.listdir(ocr_file)
        if filename.endswith('.json') or filename.endswith('.txt')
    ]

    for j in filtered_filenames:
        print(f"Processing {j}...")
        try:
            with open(os.path.join(ocr_file, j), "r", encoding="utf-8") as f:
                content = f.read()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback to literal_eval if the file uses single quotes or is just a Python dictionary string
                    data = ast.literal_eval(content)
            
            # Abbyy OCR format has a list under 'ocrContent'
            if isinstance(data, dict) and 'ocrContent' in data:
                word_coordinates = data['ocrContent']
            elif isinstance(data, list):
                # Fallback if the JSON directly contains the list
                word_coordinates = data
            else:
                print(f"Warning: Unexpected JSON structure in {j}")
                continue
                
            cou = 1
            final = {}
            for i in word_coordinates:
                # Based on the FUNSD GV target structure: {'text': word, 'bbox': [x1, y1, x2, y2]}
                # Default to 0 if bounding box coordinate is missing for some reason
                text = i.get('word', '')
                x1 = i.get('x1', 0)
                y1 = i.get('y1', 0)
                x2 = i.get('x2', 0)
                y2 = i.get('y2', 0)
                
                final[cou] = {
                    'text': text, 
                    'bbox': [x1, y1, x2, y2]
                }
                cou += 1
                
            # Format the output filename to match the image/label base name
            out_name = j
            if out_name.endswith('.txt') or out_name.endswith('.json'):
                out_name = os.path.splitext(out_name)[0]
            
            # Strip out OCR specific suffixes
            if out_name.endswith('_textAndCoordinates'):
                out_name = out_name.replace('_textAndCoordinates', '')
            elif out_name.endswith('_text'):
                out_name = out_name.replace('_text', '')
                
            out_name = out_name + '.json'
            file_name = os.path.join(all_words_path, out_name)
            
            with open(file_name, "w", encoding="utf-8") as json_file:
                json.dump(final, json_file, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"Error processing {j}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Data Generation Script for Abbyy OCR")
    parser.add_argument(
        "--root_path",
        default = '/',
        help="Root directory path",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    
    ROOT_PATH = args.root_path
    OCR_PATH = os.path.join(ROOT_PATH, 'OCR')
    gen_custom_data(ROOT_PATH, OCR_PATH)
