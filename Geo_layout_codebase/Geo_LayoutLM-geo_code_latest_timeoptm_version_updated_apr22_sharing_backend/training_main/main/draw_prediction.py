import cv2
import json
import os
import numpy as np
import random

def generate_random_color():
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color

def check_overlap(box1, box2):
    x1_start, y1_start, x1_end, y1_end = box1
    x2_start, y2_start, x2_end, y2_end = box2

    # Check for overlap
    if x1_start < x2_end and x1_end > x2_start and y1_start < y2_end and y1_end > y2_start:
        return True
    return False

def draw_rectangles(data, image_path, output_path, image_name):
    # Read the JSON data
    # with open(json_data_path, 'r') as json_file:
    #     data = json.load(json_file)

    # Load the image
    image = cv2.imread(image_path)

    drawn_keys = []  # To keep track of keys that have been drawn

    for item in data:
        pred_key = item['pred_key']
        coords_list = item['coords']
        color = generate_random_color()

        if pred_key not in drawn_keys:
            drawn_keys.append(pred_key)

            for i, coords in enumerate(coords_list):
                x_start, y_start, x_end, y_end = coords

                # Draw a rectangle and annotate the image
                cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color, 2)
                cv2.putText(image, f'{pred_key}_{i+1}', (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check for overlapping bounding boxes of the same predicted key
            for i in range(len(coords_list)):
                for j in range(i+1, len(coords_list)):
                    if check_overlap(coords_list[i], coords_list[j]):
                        print(f'Overlap detected between {pred_key}_{i+1} and {pred_key}_{j+1}')

    # Save the image with rectangles
    cv2.imwrite(os.path.join(output_path,image_name), image)

# # Provide the paths to the JSON data, the image, and the output image path
# json_data_path = 'data.json'
# image_path = 'image.jpg'
# output_image_path = 'annotated_image.jpg'

# # Call the function to draw rectangles, check for overlaps, and save the annotated image
# draw_rectangles(json_data_path, image_path, output_image_path)
