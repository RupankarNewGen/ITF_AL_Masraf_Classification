import os
import json
import cv2
from google.cloud import vision
from shapely.geometry import box
from shapely.geometry import Polygon
from tqdm import tqdm

# Function to calculate IOU
def calculate_iou(box1, box2):
    poly1 = box(*box1)
    poly2 = box(*box2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union

# Function to convert YOLO annotations to x, y coordinates
def yolo_to_coordinates(yolo_annotation, image_width, image_height):
    class_id, x_center, y_center, width, height = map(float, yolo_annotation.split())
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return class_id, (x_min, y_min, x_max, y_max)

# Initialize Google Vision client
def perform_ocr_on_image(image_path):
    """
    Performs OCR on a single image using Google Cloud Vision and dumps data to a JSON file.

    Args:
        image_path: Path to the image file.
        output_folder: Path to the folder for storing the JSON file.
    """

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    credentials_path = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/gv_key.json"

    client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)
    image = vision.Image(content=content)

    # Set context with language hints
    language = "en"
    image_context = {"language_hints": [language]}

    feature_type = "documentTextDetection"
    if feature_type == "textDetection":
        response = client.text_detection(image=image, image_context=image_context)
    elif feature_type == "documentTextDetection":
        response = client.document_text_detection(image=image, image_context=image_context)
    else:
        raise ValueError(f"Invalid feature_type: {feature_type}")

    word_coordinates = []
    all_text = ""
    for i, text in enumerate(response.text_annotations):
        if i != 0:
            # Extract word coordinates
            vertices = text.bounding_poly.vertices
            x1 = min(vertex.x for vertex in vertices)
            y1 = min(vertex.y for vertex in vertices)
            x2 = max(vertex.x for vertex in vertices)
            y2 = max(vertex.y for vertex in vertices)
            word_coordinates.append({
                "word": text.description,
                "left": x1,
                "top": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
        else:
            all_text = text.description

    data={"all_text":all_text,"word_coordinates":word_coordinates}
    return data

if __name__=="__main__":
    input_folder = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/AWB_EVL"
    image_folder = input_folder+"/Images"
    label_folder = input_folder+"/Labels"
    label_file = input_folder+"/label.txt"
    output_folder = "path_to_output_folder"
    ocr_path = input_folder+"/OCR"

    os.makedirs(ocr_path, exist_ok=True)
    # with open(label_file, "r") as f:
    #     labels = f.read().splitlines()

    for image_file in tqdm(os.listdir(image_folder)):
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + '.txt')

        directory, base_name = os.path.split(image_path)
        base_name_without_ext, _ = os.path.splitext(base_name)

        ocr_json_path = os.path.join(ocr_path,base_name_without_ext+".json")
        if not os.path.exists(ocr_json_path):
            ocr_info = perform_ocr_on_image(image_path)
            with open(ocr_json_path, 'w') as f:
                        json.dump(ocr_info, f, indent=4)

        else:
            print("OCR EXISTS")
            with open(ocr_json_path, 'r') as f:
                ocr_info = json.load(f)
        
        all_text,word_cords = ocr_info["all_text"],ocr_info["word_coordinates"]
        
        # # Load image
        # image = cv2.imread(image_path)
        # height, width, _ = image.shape

        # # Read YOLO annotations
        # with open(label_path, "r") as f:
        #     yolo_annotations = f.read().splitlines()

        # yolo_boxes = []
        # for annotation in yolo_annotations:
        #     class_id, coordinates = yolo_to_coordinates(annotation, width, height)
        #     yolo_boxes.append({
        #         "class_id": int(class_id),
        #         "label": labels[int(class_id)],
        #         "coordinates": coordinates
        #     })

