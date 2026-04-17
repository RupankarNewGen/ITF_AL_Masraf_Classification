import os
import json
import ast
import csv
from collections import Counter, defaultdict
from PIL import Image

def load_ocr_data(filepath, ocr_format): # <--- NEW: added ocr_format parameter
    """Safely loads OCR data based on the selected format ('abbyy' or 'gv')."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Try standard JSON parsing first, fallback to ast if it's a Python dict string
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = ast.literal_eval(content)
            
            # <--- NEW: Select the correct root key based on format
            if ocr_format.lower() == 'abbyy':
                return data.get('ocrContent', [])
            elif ocr_format.lower() == 'gv':
                return data.get('word_coordinates', [])
            else:
                print(f"  ⚠️ Unknown OCR format '{ocr_format}'. Please use 'abbyy' or 'gv'.")
                return []
                
    except Exception as e:
        print(f"  ⚠️ Error reading OCR file {os.path.basename(filepath)}: {e}")
        return []

def get_text_from_ocr(yolo_bbox, img_width, img_height, ocr_data):
    """
    Converts YOLO normalized bbox to absolute pixels and extracts overlapping OCR text.
    """
    x_center_norm, y_center_norm, w_norm, h_norm = yolo_bbox
    
    # 1. Convert YOLO to Absolute Pixels
    w_abs = w_norm * img_width
    h_abs = h_norm * img_height
    x_center_abs = x_center_norm * img_width
    y_center_abs = y_center_norm * img_height
    
    yolo_x1 = x_center_abs - (w_abs / 2)
    yolo_y1 = y_center_abs - (h_abs / 2)
    yolo_x2 = x_center_abs + (w_abs / 2)
    yolo_y2 = y_center_abs + (h_abs / 2)
    
    extracted_words = []
    
    # 2. Check each word for intersection with the YOLO box
    for item in ocr_data:
        word_x1, word_y1 = item.get('x1'), item.get('y1')
        word_x2, word_y2 = item.get('x2'), item.get('y2')
        word_text = item.get('word', '')
        
        if word_x1 is None or not word_text:
            continue
            
        # Calculate intersection area
        inter_x_min = max(yolo_x1, word_x1)
        inter_y_min = max(yolo_y1, word_y1)
        inter_x_max = min(yolo_x2, word_x2)
        inter_y_max = min(yolo_y2, word_y2)
        
        if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            word_area = (word_x2 - word_x1) * (word_y2 - word_y1)
            
            # If at least 40% of the OCR word is inside the YOLO box, keep it
            if word_area > 0 and (inter_area / word_area) >= 0.40:
                # Store the Y and X coordinates to sort the words properly later
                extracted_words.append((word_y1, word_x1, word_text))
    
    # 3. Sort words naturally (top-to-bottom, left-to-right) and join them
    # Grouping Y coordinates by 10 pixels to handle slight misalignments in the same line
    extracted_words.sort(key=lambda w: (w[0] // 10, w[1])) 
    
    final_text = " ".join([w[2] for w in extracted_words])
    return final_text.strip()


def analyze_label_distribution(labels_folder, images_folder, classes_filepath, output_filepath, ocr_folder, output_json_path, output_csv_path, ocr_format): # <--- NEW: added ocr_format parameter
    print("Loading class names...")
    class_names = []
    if os.path.exists(classes_filepath):
        with open(classes_filepath, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        print(f"⚠️ Classes file not found at: {classes_filepath}")
        return

    class_counts = Counter()
    total_valid_images = 0  
    class_values_dict = defaultdict(list)
    valid_image_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

    print(f"Scanning label files in: {labels_folder}...")
    
    for filename in os.listdir(labels_folder):
        if filename.endswith('.txt') and filename != os.path.basename(classes_filepath):
            label_filepath = os.path.join(labels_folder, filename)
            base_name = filename.replace('.txt', '')
            
            # --- 1. FIND AND LOAD THE MATCHING OCR FILE FIRST ---
            ocr_filename = f"{base_name}_text.txt"
            ocr_filepath = os.path.join(ocr_folder, ocr_filename)
            
            # THE SKIP LOGIC: If OCR doesn't exist, print and move to the next file instantly
            if not os.path.exists(ocr_filepath):
                print(f"⚠️ OCR is not found for image: {base_name}. Skipping...")
                continue
                
            ocr_data = load_ocr_data(ocr_filepath, ocr_format) # <--- NEW: Pass ocr_format to the loader
            
            # --- 2. FIND THE MATCHING IMAGE TO GET DIMENSIONS ---
            img_width, img_height, final_img_path = None, None, None
            for ext in valid_image_exts:
                img_path = os.path.join(images_folder, base_name + ext)
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img:
                            img_width, img_height = img.size
                            final_img_path = img_path 
                        break
                    except Exception as e:
                        print(f"  ⚠️ Error opening image {img_path}: {e}")
            
            if img_width is None:
                print(f"  ⚠️ Missing or unreadable image for {base_name}. Skipping...")
                continue

            # If it passed OCR and Image checks, it's a valid image
            total_valid_images += 1
            classes_in_current_image = set() 

            # --- 3. PROCESS YOLO BOXES ---
            with open(label_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5: # Ensure we have class + 4 coordinates
                        try:
                            class_id = int(parts[0])
                            
                            # Add to our per-image tracker
                            classes_in_current_image.add(class_id)
                            
                            # Get class name safely
                            if 0 <= class_id < len(class_names):
                                name = class_names[class_id]
                            else:
                                name = f"Unknown_Class_{class_id}"
                            
                            # Extract Text
                            yolo_bbox = [float(p) for p in parts[1:5]]
                            extracted_text = get_text_from_ocr(yolo_bbox, img_width, img_height, ocr_data)
                            
                            # Store as a dictionary with value and image path
                            class_values_dict[name].append({
                                "value": extracted_text,
                                "image_path": final_img_path
                            })
                                
                        except ValueError:
                            continue
            
            for cid in classes_in_current_image:
                class_counts[cid] += 1

    # --- Generate the standard Text Report ---
    print("\nCalculating statistics...")
    with open(output_filepath, 'w', encoding='utf-8') as out_f:
        out_f.write("=== LABEL DISTRIBUTION REPORT (PER IMAGE) ===\n")
        out_f.write(f"Total Valid Images Processed: {total_valid_images}\n")
        out_f.write("-" * 55 + "\n")
        out_f.write(f"{'Class Name':<30} | {'Image Count':<11} | {'Percentage'}\n")
        out_f.write("-" * 55 + "\n")

        if total_valid_images == 0:
            out_f.write("No valid labels found in the provided folder.\n")
        else:
            for class_id, count in class_counts.most_common():
                percentage = (count / total_valid_images) * 100
                if 0 <= class_id < len(class_names):
                    name = class_names[class_id]
                else:
                    name = f"Unknown_Class_{class_id}"
                
                out_f.write(f"{name:<30} | {count:<11} | {percentage:.2f}%\n")

    # --- Generate the JSON Output ---
    print("Saving extracted values to JSON...")
    with open(output_json_path, 'w', encoding='utf-8') as json_f:
        json.dump(class_values_dict, json_f, indent=4, ensure_ascii=False)

    # --- Generate the CSV Output ---
    print("Saving extracted values to CSV...")
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(['Label Name', 'Value', 'Image Name']) # Write headers
        
        for label_name, items in class_values_dict.items():
            for item in items:
                # Extract just the image name (e.g., image1.jpg) from the full path
                image_name = os.path.basename(item['image_path'])
                writer.writerow([label_name, item['value'], image_name])

    print(f"\n✅ Analysis complete! Report saved to: {output_filepath}")
    print(f"✅ JSON values saved to: {output_json_path}")
    print(f"✅ CSV values saved to: {output_csv_path}")


if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    LABELS_FOLDER = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/old_tf_data/CS_root/Labels"
    IMAGES_FOLDER = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/old_tf_data/CS_root/Images"
    OCR_FOLDER    = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/old_tf_data/CS_root/OCR"
    CLASSES_FILE  = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/old_tf_data/CS_root/label.txt"
    
    OUTPUT_REPORT = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/old_tf_data/CS_root/label_distribution_report.txt"
    OUTPUT_JSON   = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/old_tf_data/CS_root/extracted_values.json"
    OUTPUT_CSV    = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/old_tf_data/CS_root/extracted_values.csv"
    
    # <--- NEW: Specify the OCR format here! ('abbyy' or 'gv')
    OCR_FORMAT    = "gv" 
    
    analyze_label_distribution(
        labels_folder=LABELS_FOLDER, 
        images_folder=IMAGES_FOLDER, 
        classes_filepath=CLASSES_FILE, 
        output_filepath=OUTPUT_REPORT, 
        ocr_folder=OCR_FOLDER, 
        output_json_path=OUTPUT_JSON,
        output_csv_path=OUTPUT_CSV,
        ocr_format=OCR_FORMAT # <--- NEW: Passed into function
    )