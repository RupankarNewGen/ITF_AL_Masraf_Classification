#!/usr/bin/env python3
"""
Visualization script to compare Ground Truth (GT) annotations with Model Predictions.

Draws:
- GREEN boxes: Ground Truth zones from zone_output/ directories
- RED boxes: Model predictions from Test_jsons/ results
"""

import os
import json
import cv2
import re
import pandas as pd
from pathlib import Path


# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data"
CLASS_ROOT = os.path.join(DATA_DIR, "Test_Data_latest")
JSON_INPUT_DIR = os.path.join(DATA_DIR, "Test_jsons")
ODS_FILE = os.path.join(DATA_DIR, "Final Results Sheet.ods")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "visualizations_gt_vs_pred")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# UTILITIES
# ============================================================================

def normalize_doc_name(name: str):
    """Normalize document names to handle various naming conventions."""
    name = name.replace(".json", "").strip()
    name = re.sub(r"(_page_\d+|__\d+).*$", "", name)
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)
    name = re.sub(r"_ver_\d+$", "", name)
    name = re.sub(r"_\d+\s*Pages?.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_debug.*$", "", name)
    return name.strip()


def load_gt_classes(ods_file):
    """Load GT document classes from ODS file."""
    try:
        df = pd.read_excel(ods_file, engine="odf")
    except Exception as e:
        print(f"❌ Error loading ODS file: {e}")
        return {}

    gt_map = {}
    
    for col in df.columns[1:]:
        raw_name = col.strip()
        norm_name = normalize_doc_name(raw_name)
        
        if norm_name not in gt_map:
            gt_map[norm_name] = {}

        for _, row in df.iterrows():
            page_str = str(row["Page Number"]).strip()
            match = re.search(r'(\d+)', page_str)
            if not match:
                continue
            
            page = int(match.group(1))
            gt_class = row[col]
            
            if pd.notna(gt_class):
                gt_map[norm_name][page] = str(gt_class).strip()

    print(f"✅ Loaded GT Classes from ODS:")
    for doc_name, pages in sorted(gt_map.items())[:5]:
        print(f"   {doc_name}: {len(pages)} pages")
    
    return gt_map


def get_gt_class(gt_classes, doc_name, page_num):
    """Case-insensitive GT class lookup."""
    if doc_name in gt_classes:
        return gt_classes[doc_name].get(page_num, "Unknown")
    
    doc_name_lower = doc_name.lower()
    for key in gt_classes.keys():
        if key.lower() == doc_name_lower:
            return gt_classes[key].get(page_num, "Unknown")
    
    return "Unknown"


def load_gt_bboxes(class_root, doc_name, page_num):
    """
    Load GT bounding boxes from zone_output JSON files.
    Structure: { "field_name": { "words": [ {"word": "...", "bbox": [x1,y1,x2,y2]}, ...] } }
    """
    target_page = f"_page_{page_num}.json"
    gt_bboxes = {}  # { field_label: [[x1,y1,x2,y2], ...] }
    
    for cls in os.listdir(class_root):
        cls_path = os.path.join(class_root, cls)
        zone_output_dir = os.path.join(cls_path, "zone_output")
        
        if not os.path.isdir(zone_output_dir):
            continue
        
        for file in os.listdir(zone_output_dir):
            if not file.endswith(target_page):
                continue
            
            # Normalize both names for comparison
            file_normalized = normalize_doc_name(file).lower()
            doc_normalized = normalize_doc_name(doc_name).lower()
            
            if file_normalized == doc_normalized or file_normalized.startswith(doc_normalized):
                file_path = os.path.join(zone_output_dir, file)
                
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    
                    for field_label, details in data.items():
                        words = details.get("words", [])
                        bboxes = []
                        
                        for word_obj in words:
                            bbox = word_obj.get("bbox")
                            if bbox and len(bbox) == 4:
                                bboxes.append(bbox)
                        
                        if bboxes:
                            gt_bboxes[field_label] = bboxes
                    
                    if gt_bboxes:
                        print(f"  ✅ Found GT bboxes: {len(gt_bboxes)} fields with {sum(len(b) for b in gt_bboxes.values())} total boxes")
                    return gt_bboxes
                    
                except Exception as e:
                    print(f"  ⚠️ Error loading GT bboxes from {file}: {e}")
                    return {}
    
    return {}


def load_pred_bboxes(pred_data):
    """
    Load prediction bounding boxes from model result JSON.
    Structure: { "field_name": { "coordinate": [[x1,y1,x2,y2], ...] } }
    """
    pred_bboxes = {}  # { field_label: [[x1,y1,x2,y2], ...] }
    
    extraction_result = pred_data.get("extraction_result", {})
    
    for field_label, details in extraction_result.items():
        if not isinstance(details, dict):
            continue
        
        coords = details.get("coordinate", [])
        if coords:
            pred_bboxes[field_label] = coords
    
    return pred_bboxes


def find_image_file(class_root, doc_name, page_num):
    """Search for the image file in image_data directories."""
    for cls in os.listdir(class_root):
        cls_path = os.path.join(class_root, cls)
        img_dir = os.path.join(cls_path, "image_data")
        
        if not os.path.isdir(img_dir):
            continue
        
        # Try various formats
        for ext in ['.png', '.tif', '.tiff', '.jpg', '.jpeg']:
            for fmt in [f"{doc_name}_{page_num}{ext}", f"{doc_name}_page_{page_num}{ext}"]:
                potential_path = os.path.join(img_dir, fmt)
                if os.path.exists(potential_path):
                    return potential_path
    
    return None


def draw_boxes(img, bboxes_dict, label_color, label_text):
    """
    Draw bounding boxes on image.
    
    Args:
        img: numpy array (image)
        bboxes_dict: { field_label: [[x1,y1,x2,y2], ...], ... }
        label_color: (B, G, R) tuple for box color
        label_text: prefix for label (e.g., "GT:" or "PRED:")
    """
    for field_label, bbox_list in bboxes_dict.items():
        for bbox in bbox_list:
            if len(bbox) == 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), label_color, 2)
                
                # Draw label
                label = f"{label_text} {field_label}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                
                label_x = x1
                label_y = y1 - 5 if y1 > 20 else y1 + 20
                
                # White background for label
                cv2.rectangle(img, (label_x, label_y - text_size[1] - 4), 
                            (label_x + text_size[0] + 4, label_y + 4), (255, 255, 255), -1)
                
                # Black text
                cv2.putText(img, label, (label_x + 2, label_y), font, font_scale, 
                          (0, 0, 0), thickness)


def visualize_document(pred_json_file, gt_classes, class_root):
    """Visualize GT and predictions for a single prediction JSON file."""
    
    try:
        with open(pred_json_file, "r") as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading prediction JSON: {e}")
        return False
    
    # Extract document name from JSON filename
    json_filename = os.path.basename(pred_json_file)
    doc_name = normalize_doc_name(json_filename)
    
    result = pred_data.get("result", {})
    
    for page_key, page_data in result.items():
        # Extract page number from key like "input_pdf_file__0.png"
        match = re.search(r"__(\d+)", page_key)
        if not match:
            continue
        
        pred_page_index = int(match.group(1))
        gt_page_num = pred_page_index + 1  # Convert to 1-indexed
        
        # Get image
        image_path = find_image_file(class_root, doc_name, gt_page_num)
        if not image_path:
            print(f"⚠️ Image not found for {doc_name} page {gt_page_num}")
            continue
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Failed to read image: {image_path}")
                continue
        except Exception as e:
            print(f"❌ Error reading image: {e}")
            continue
        
        # Get GT class
        gt_class = get_gt_class(gt_classes, doc_name, gt_page_num)
        
        # Load GT bboxes (GREEN)
        gt_bboxes = load_gt_bboxes(class_root, doc_name, gt_page_num)
        
        # Load prediction bboxes (RED)
        pred_bboxes = load_pred_bboxes(page_data)
        
        # Draw bboxes
        draw_boxes(img, gt_bboxes, (0, 255, 0), "GT")      # Green
        draw_boxes(img, pred_bboxes, (0, 0, 255), "PRED")  # Red
        
        # Add header info
        pred_class = page_data.get("predicted_class", "?")
        header = f"GT Class: {gt_class} | Predicted: {pred_class}"
        cv2.putText(img, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Save annotated image
        output_name = f"{doc_name}_page_{gt_page_num}.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        cv2.imwrite(output_path, img)
        print(f"✅ Saved: {output_name}")
    
    return True


def main():
    print("=" * 80)
    print("GT vs Predictions Visualization")
    print("=" * 80)
    
    # Load GT classes
    gt_classes = load_gt_classes(ODS_FILE)
    if not gt_classes:
        print("❌ No GT classes loaded. Exiting.")
        return
    
    # Process each prediction JSON file
    json_files = list(Path(JSON_INPUT_DIR).glob("*.json"))
    print(f"\n📁 Found {len(json_files)} prediction JSON files\n")
    
    count = 0
    for json_file in sorted(json_files):  # Process ALL files
        print(f"\n📄 Processing: {json_file.name}")
        if visualize_document(str(json_file), gt_classes, CLASS_ROOT):
            count += 1
    
    print(f"\n✅ Done! Generated visualizations in: {OUTPUT_DIR}")
    print(f"   Total files processed: {count}")
    
    # Count generated images
    total_images = len(list(Path(OUTPUT_DIR).glob("*.png")))
    print(f"   Total images generated: {total_images}")


if __name__ == "__main__":
    main()
