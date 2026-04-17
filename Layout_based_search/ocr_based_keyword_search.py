import os
import shutil
import json

def filter_ocr_and_images(input_folder, image_input_folder, output_folder, keyword_list):
    """
    Parses OCR JSON files, finds matching keywords, and copies both 
    the .txt file and its corresponding image to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']
    
    files_processed = 0
    matches_found = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        continue
                        
                    data = json.loads(content)
                    
                    if isinstance(data, dict) and "word_coordinates" in data:
                        word_items = data["word_coordinates"]
                    elif isinstance(data, list):
                        word_items = data
                    else:
                        continue

                    full_text = " ".join([str(item['word']) for item in word_items if isinstance(item, dict) and 'word' in item])
                    
                    print(f"\n" + "="*50)
                    print(f"FILE: {filename}")
                    
                    matched_keywords = [kw for kw in keyword_list if kw.lower() in full_text.lower()]
                    
                    if matched_keywords:
                        print(f"SUCCESS: Matching keywords: {', '.join(matched_keywords)}")
                        
                        # 1. Copy the OCR text file
                        shutil.copy(file_path, os.path.join(output_folder, filename))
                        
                        # 2. Logic to find and copy the corresponding image
                        # Remove '_text.txt' to get 'Covering_Schedule_0_page_0'
                        base_name = filename.replace("_text.txt", "")
                        
                        image_found = False
                        for ext in image_extensions:
                            potential_img_name = base_name + ext
                            potential_img_path = os.path.join(image_input_folder, potential_img_name)
                            
                            if os.path.exists(potential_img_path):
                                shutil.copy(potential_img_path, os.path.join(output_folder, potential_img_name))
                                print(f"IMAGE MATCH: Found and copied {potential_img_name}")
                                image_found = True
                                break # Stop looking once one extension matches
                        
                        if not image_found:
                            print(f"IMAGE WARNING: No matching image found for {base_name}")
                            
                        matches_found += 1
                    else:
                        print("RESULT: No matching keyword found.")
                
                files_processed += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\n" + "X"*40)
    print(f"FINAL SUMMARY")
    print(f"Total OCR files processed: {files_processed}")
    print(f"Total Matches (Text + Image): {matches_found}")
    print(f"X"*40)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    input_ocr_dir = "/home/lpt6964/Downloads/ITF_utils/layout_segementation.py/CS_OCR"
    input_image_dir = "/home/lpt6964/Downloads/ITF_utils/layout_segementation.py/CS_Images"
    output_dir = "/home/lpt6964/Downloads/ITF_utils/layout_segementation.py/CS_matched_ocr_files_17"
    
    # search_keywords = ["RAKBANK"]
    search_keywords = ["first abu dhabi bank"]
    # ---------------------

    filter_ocr_and_images(input_ocr_dir, input_image_dir, output_dir, search_keywords)