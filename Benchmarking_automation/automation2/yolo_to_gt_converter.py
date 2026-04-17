import os
import ast
import json
from PIL import Image

def get_intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def get_box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def process_file(yolo_file_path, ocr_file_path, image_path, classes_map, output_file_path):
    if not os.path.exists(yolo_file_path):
        return
    if not os.path.exists(ocr_file_path):
        print(f"Skipping {os.path.basename(yolo_file_path)}: Missing OCR file")
        return
    if not os.path.exists(image_path):
        print(f"Skipping {os.path.basename(yolo_file_path)}: Missing image file")
        return
    
    with Image.open(image_path) as img:
        img_w, img_h = img.size
        
    with open(yolo_file_path, 'r') as f:
        yolo_lines = f.readlines()
        
    with open(ocr_file_path, 'r') as f:
        ocr_content_str = f.read()
        
    try:
        ocr_data = ast.literal_eval(ocr_content_str)
        ocr_words = ocr_data.get('ocrContent', [])
    except Exception as e:
        print(f"Error parsing OCR file {os.path.basename(ocr_file_path)}: {e}")
        return
        
    output_data = {}
    
    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        class_idx = int(parts[0])
        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h
        
        yolo_x1 = cx - w / 2
        yolo_y1 = cy - h / 2
        yolo_x2 = cx + w / 2
        yolo_y2 = cy + h / 2
        yolo_box = [yolo_x1, yolo_y1, yolo_x2, yolo_y2]
        
        class_name = classes_map.get(class_idx, f"class_{class_idx}")
        
        intersecting_words = []
        for word_info in ocr_words:
            ocr_box = [word_info['x1'], word_info['y1'], word_info['x2'], word_info['y2']]
            intersection = get_intersection_area(yolo_box, ocr_box)
            ocr_area = get_box_area(ocr_box)
            if ocr_area > 0 and (intersection / ocr_area) >= 0.5:
                # Add word text and its box
                intersecting_words.append((word_info.get('word', ''), ocr_box))
                
        # Sort words primarily by top y coordinate (allowing slight variations for same line)
        # and secondarily by left x coordinate
        # Let's say a line tolerance of 10 pixels
        intersecting_words.sort(key=lambda item: (item[1][1] // 10, item[1][0]))
        
        combined_text = " ".join([word for word, box in intersecting_words if word])
        
        rounded_yolo_box = [int(yolo_x1), int(yolo_y1), int(yolo_x2), int(yolo_y2)]
        
        if class_name not in output_data:
            output_data[class_name] = []
            
        output_data[class_name].append([combined_text, rounded_yolo_box])
        
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Processed: {os.path.basename(output_file_path)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert YOLO labels to ground truth.")
    parser.add_argument("--yolo_dir", required=True, help="Path to YOLO labels directory")
    parser.add_argument("--ocr_dir", required=True, help="Path to OCR directory")
    parser.add_argument("--img_dir", required=True, help="Path to the images directory")
    parser.add_argument("--classes_file", required=True, help="Path to the classes.txt file for the specific class")
    parser.add_argument("--output_dir", required=True, help="Where to save the GT txt files")
    args = parser.parse_args()
    
    yolo_dir = args.yolo_dir
    ocr_dir = args.ocr_dir
    img_dir = args.img_dir
    classes_file = args.classes_file
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    classes_map = {}
    with open(classes_file, 'r') as f:
        for idx, line in enumerate(f):
            label = line.strip()
            if label:
                classes_map[idx] = label
                
    yolo_files = [f for f in os.listdir(yolo_dir) if f.endswith('.txt')]
    for yf in yolo_files:
        base_name = os.path.splitext(yf)[0]
        
        img_path = None
        for ext in ['.jpeg', '.jpg', '.png']:
            p = os.path.join(img_dir, base_name + ext)
            if os.path.exists(p):
                img_path = p
                break
                
        if not img_path:
            # default fallback
            img_path = os.path.join(img_dir, base_name + '.jpeg')
            
        ocr_file_path = os.path.join(ocr_dir, f"{base_name}_textAndCoordinates.txt")
        yolo_file_path = os.path.join(yolo_dir, yf)
        output_file_path = os.path.join(output_dir, f"{base_name}_labels.txt")
        
        process_file(yolo_file_path, ocr_file_path, img_path, classes_map, output_file_path)

if __name__ == '__main__':
    main()
