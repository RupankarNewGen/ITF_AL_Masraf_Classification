import os
import json
import argparse
import csv
from PIL import Image

CLASSES_FILE_MAP = {
    "BOE": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations_Test/Test_BOE/classes.txt",
    "BOL": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations_Test/Test_BOL/classes.txt",
    "CI": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations_Test/Test_CI/classes.txt",
    "COO": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations_Test/Test_COO/classes.txt",
    "CS": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations_Test/Test_CS/classes.txt",
    "PL": "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Annotations_Test/Test_PL/classes.txt"
}

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None, None

def get_intersection_area(box1, box2):
    # box: [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def get_box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def build_label_and_image_index(search_folders):
    """
    Scans the provided directories to build an index mapping 
    base filename to its .txt label path and .png/.jpeg image path.
    """
    valid_img_exts = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    label_map = {}
    image_map = {}
    
    for folder in search_folders:
        if not os.path.exists(folder):
            continue
        for root, _, files in os.walk(folder):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                base_name = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                
                if ext == '.txt' and file != 'classes.txt':
                    label_map[base_name] = full_path
                elif ext in valid_img_exts:
                    image_map[base_name] = full_path
                    
    return label_map, image_map

def load_classes(classes_file_path):
    sign_id = None
    stamp_id = None
    
    if os.path.exists(classes_file_path):
        with open(classes_file_path, 'r') as f:
            for idx, line in enumerate(f):
                class_name = line.strip().lower()
                if "sign" in class_name:
                    sign_id = idx
                if "stamp" in class_name:
                    stamp_id = idx
    return sign_id, stamp_id

def process_file(json_file_path, label_map, image_map, classes_file_map, output_rows):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    result_dict = data.get("result", {})
    json_basename = os.path.splitext(os.path.basename(json_file_path))[0]
    
    # Cache loaded class IDs per predicted_class to avoid re-reading files
    class_id_cache = {}
    valid_classes = set(classes_file_map.keys())
    
    for pdf_key, page_data in result_dict.items():
        # Get predicted class for this page; skip if not in classes_file_map
        predicted_class = page_data.get("predicted_class", "")
        if predicted_class not in valid_classes:
            continue
        
        # Load sign/stamp IDs for this predicted class (cached)
        if predicted_class not in class_id_cache:
            class_id_cache[predicted_class] = load_classes(classes_file_map[predicted_class])
        sign_id, stamp_id = class_id_cache[predicted_class]
        
        # Map input_pdf_file__X -> DocumentName_page_X+1
        if "input_pdf_file__" in pdf_key:
            try:
                page_idx_str = pdf_key.split("input_pdf_file__")[1].split(".")[0]
                page_num = int(page_idx_str) + 1
                matching_name = f"{json_basename}_page_{page_num}"
            except Exception:
                matching_name = pdf_key.split(".")[0]
        else:
            matching_name = pdf_key.split(".")[0]
            
        # Try to find ground truth text file
        label_file = label_map.get(matching_name)
        if not label_file:
            continue
            
        # Get physical image for dimensions
        image_file = image_map.get(matching_name)
        img_w, img_h = None, None
        if image_file:
            img_w, img_h = get_image_size(image_file)
            
        gt_signs = []
        gt_stamps = []
        
        # Parse YOLO labels
        with open(label_file, 'r') as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    if img_w and img_h:
                        x1 = (cx - w / 2) * img_w
                        y1 = (cy - h / 2) * img_h
                        x2 = (cx + w / 2) * img_w
                        y2 = (cy + h / 2) * img_h
                        box = [x1, y1, x2, y2]
                    else:
                        # Fallback to normalized checking if image isn't found
                        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                        
                    if cls_id == sign_id:
                        gt_signs.append(box)
                    elif cls_id == stamp_id:
                        gt_stamps.append(box)

        # Get Pred Boxes (only from extraction_result)
        pred_signs = []
        pred_stamps = []
        
        ex_res = page_data.get("extraction_result", {})
        
        if "is_signed" in ex_res and "coordinate" in ex_res["is_signed"]:
            pred_signs.extend(ex_res["is_signed"]["coordinate"])
                
        if "is_stamp" in ex_res and "coordinate" in ex_res["is_stamp"]:
            pred_stamps.extend(ex_res["is_stamp"]["coordinate"])

        # Match GT to Pred with greedy one-to-one matching (highest intersection first)
        def match_boxes(gt_boxes, pred_boxes, is_normalized=False):
            matched_pred_indices = set()
            detected_count = 0
            for gt in gt_boxes:
                best_inter = 0
                best_pred_idx = -1
                for pred_idx, pred in enumerate(pred_boxes):
                    if pred_idx in matched_pred_indices:
                        continue
                    inter = get_intersection_area(gt, pred)
                    if inter > best_inter:
                        best_inter = inter
                        best_pred_idx = pred_idx
                if best_inter > 0 and best_pred_idx >= 0:
                    detected_count += 1
                    matched_pred_indices.add(best_pred_idx)
            return detected_count

        has_img = (img_w is not None)
        # If no image was found, GT boxes are normalized. Predictions are absolute. Intersection will likely fail 
        # unless prediction boxes are also mapped back to normalized. We print a warning.
        if not has_img and (gt_signs or gt_stamps):
            print(f"Warning: Image for {matching_name} not found to convert YOLO scale. Intersections may be 0.")

        detected_signs = match_boxes(gt_signs, pred_signs, is_normalized=not has_img)
        detected_stamps = match_boxes(gt_stamps, pred_stamps, is_normalized=not has_img)

        # Output rows mechanism
        # The user requested: image_name, sign_detected, stamp_detected
        # By iterating to output the counts:
        if gt_signs or gt_stamps:
            output_rows.append({
                "image_name": matching_name,
                "sign_detected": detected_signs,
                "stamp_detected": detected_stamps
            })


def main():
    parser = argparse.ArgumentParser(description="Benchmarking Sign and Stamp Detection")
    parser.add_argument("--json_dir", required=True, help="Directory containing model prediction JSONs")
    parser.add_argument("--labels_dirs", nargs='+', required=True, help="List of folders containing ground truth labels and images")
    parser.add_argument("--output_csv", required=True, help="Path to output the benchmark report")
    
    args = parser.parse_args()
    
    classes_file_map = CLASSES_FILE_MAP
    print(f"Loaded classes_file_map for classes: {list(classes_file_map.keys())}")
        
    print("Indexing label files and physical images...")
    label_map, image_map = build_label_and_image_index(args.labels_dirs)
    print(f"Indexed {len(label_map)} labels and {len(image_map)} images.")
    
    output_rows = []
    
    for json_file in os.listdir(args.json_dir):
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join(args.json_dir, json_file)
        process_file(json_path, label_map, image_map, classes_file_map, output_rows)
        
    # Write report
    with open(args.output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "sign_detected", "stamp_detected"])
        writer.writeheader()
        writer.writerows(output_rows)
        
    print(f"\n✅ Benchmarking completed. Results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
