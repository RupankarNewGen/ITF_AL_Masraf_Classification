

import os
import json
import time
from pathlib import Path
from google.cloud import vision


# ------------------- Hardcoded paths -------------------
INPUT_FOLDER = "/home/lpt6964/Downloads/ITF_utils/temp/image"
OUTPUT_FOLDER = "/home/lpt6964/Downloads/ITF_utils/temp/ocr"

# ------------------- Load service account JSON -------------------
config_path = os.path.join(os.path.dirname(__file__), "gv_config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError("gv_config.json not found in current directory.")

with open(config_path, "r") as f:
    service_account_info = json.load(f)

language_hint = "en"
feature_type = "documentTextDetection"

# ------------------- Create output folder -------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------- Supported image extensions -------------------
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")


def process_image(image_path):
    """Perform OCR on a single image using Google Vision API"""
    client = vision.ImageAnnotatorClient.from_service_account_info(service_account_info)

    with open(image_path, 'rb') as img_file:
        content = img_file.read()
    image = vision.Image(content=content)
    image_context = {"language_hints": [language_hint]}

    if feature_type == "textDetection":
        response = client.text_detection(image=image, image_context=image_context)
    else:
        response = client.document_text_detection(image=image, image_context=image_context)

    # Full text
    full_text = response.text_annotations[0].description if response.text_annotations else ""

    # Word-level coordinates
    word_coordinates = []
    for i, text in enumerate(response.text_annotations):
        if i == 0:
            continue
        vertices = text.bounding_poly.vertices
        x1 = min(v.x for v in vertices)
        y1 = min(v.y for v in vertices)
        x2 = max(v.x for v in vertices)
        y2 = max(v.y for v in vertices)
        word_coordinates.append({
            "word": text.description,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1
        })

    return full_text, word_coordinates


# ------------------- Process all images -------------------
for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(VALID_EXTENSIONS):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)
    base_name = Path(filename).stem

    print(f"\nProcessing file: {filename}")
    start_time = time.time()

    try:
        full_text, word_coords = process_image(image_path)

        # Save outputs
        text_output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_alltext.txt")
        coord_output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_word_coordinates.txt")

        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        with open(coord_output_path, "w", encoding="utf-8") as f:
            json.dump(word_coords, f, indent=2)

        elapsed = time.time() - start_time
        print(f"✅ Completed: {filename} | Time taken: {elapsed:.2f} seconds")

    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")






