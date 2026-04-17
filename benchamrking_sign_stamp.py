import os
import json
import argparse
import csv
from PIL import Image

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
    class_map = {}
    sign_id = None
    stamp_id = None
    
    if os.path.exists(classes_file_path):
        with open(classes_file_path, 'r') as f:
            for idx, line in enumerate(f):
                class_name = line.strip().lower()
                class_map[idx] = class_name
                if "sign" in class_name:
                    sign_id = idx
                if "stamp" in class_name:
                    stamp_id = idx
    return sign_id, stamp_id

def process_file(json_file_path, label_map, image_map, sign_id, stamp_id, output_rows):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    result_dict = data.get("result", {})
    json_basename = os.path.splitext(os.path.basename(json_file_path))[0]
    
    for pdf_key, page_data in result_dict.items():
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

        # Get Pred Boxes
        pred_signs = []
        pred_stamps = []
        
        ex_res = page_data.get("extraction_result", {})
        
        if "is_signed" in ex_res and "coordinate" in ex_res["is_signed"]:
            pred_signs.extend(ex_res["is_signed"]["coordinate"])
        if "docAutoSignDetection" in page_data:
            b = page_data["docAutoSignDetection"].get("keys_bboxes", {}).get("sign", [])
            if len(b) > 0 and isinstance(b[0], list):
                pred_signs.extend(b)
            elif b:
                pred_signs.append(b)
                
        if "is_stamp" in ex_res and "coordinate" in ex_res["is_stamp"]:
            pred_stamps.extend(ex_res["is_stamp"]["coordinate"])
        if "docAutoStampDetection" in page_data:
            b = page_data["docAutoStampDetection"].get("keys_bboxes", {}).get("stamp", [])
            if len(b) > 0 and isinstance(b[0], list):
                pred_stamps.extend(b)
            elif b:
                pred_stamps.append(b)

        # Intersect function
        def match_boxes(gt_boxes, pred_boxes, is_normalized=False):
            detected_count = 0
            for gt in gt_boxes:
                best_iou = 0
                for pred in pred_boxes:
                    # If GT is normalized but Pred is absolute, we have an issue.
                    # As fallback, just assume intersection > 0 is found mechanically 
                    # if the ranges overlap significantly or if we have img_w/img_h.
                    if is_normalized:
                        pass # Cannot accurately match absolute pred with norm GT without img dims
                        
                    inter = get_intersection_area(gt, pred)
                    if inter > 0:
                        best_iou = inter
                        
                # Just requires > 0 intersection
                if best_iou > 0:
                    detected_count += 1
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
    parser.add_argument("--classes_file", required=True, help="Path to classes.txt mapping file")
    parser.add_argument("--output_csv", required=True, help="Path to output the benchmark report")
    
    args = parser.parse_args()
    
    sign_id, stamp_id = load_classes(args.classes_file)
    print(f"Loaded class map. Sign ID: {sign_id}, Stamp ID: {stamp_id}")
    
    if sign_id is None and stamp_id is None:
        print("Warning: Neither 'sign' nor 'stamp' found in classes file.")
        
    print("Indexing label files and physical images...")
    label_map, image_map = build_label_and_image_index(args.labels_dirs)
    print(f"Indexed {len(label_map)} labels and {len(image_map)} images.")
    
    output_rows = []
    
    for json_file in os.listdir(args.json_dir):
        if not json_file.endswith('.json'):
            continue
        json_path = os.path.join(args.json_dir, json_file)
        process_file(json_path, label_map, image_map, sign_id, stamp_id, output_rows)
        
    # Write report
    with open(args.output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "sign_detected", "stamp_detected"])
        writer.writeheader()
        writer.writerows(output_rows)
        
    print(f"\n✅ Benchmarking completed. Results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
