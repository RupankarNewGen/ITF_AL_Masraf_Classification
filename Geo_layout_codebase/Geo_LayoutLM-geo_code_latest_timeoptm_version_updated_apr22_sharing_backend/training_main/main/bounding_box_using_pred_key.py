import cv2
import json
import os
import numpy as np
import random

def generate_random_color():
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color

def draw_rectangles(data, image_path, output_path, image_name):
    print(f'image path: {image_path}')
    image = cv2.imread(image_path)
    coords_lst= []

    for item in data:
        pred_key = item['pred_key']
        # pred_key=pred_key.replace('B-','')
        # pred_key=pred_key.replace('I-','')  
        coords = item['coords']
        color = generate_random_color()

        x_start, y_start, x_end, y_end = coords
        if pred_key not in coords_lst:
            coords_lst.append(pred_key)
        # if pred_key not in coords_lst:
        # Draw a rectangle around the predicted key with a random color
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color, 2)
            cv2.putText(image, pred_key, (x_end, y_end - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            pass
    # print(coords_lst)
    # Save the image with rectangles
    print(type(image))
    cv2.imwrite(os.path.join(output_path,image_name ), image)

# # Provide the paths to the JSON data, the image, and the output image path
# json_data_path = 'custom_result/custom_trial/Certificate_Of_Origin_142_page_1_tagging.json'
# image_path = 'custom_result/custom_trial/vis/Certificate_Of_Origin_142_page_1_linking.png'
# output_image_path = 'annotated_image.jpg'

# # Call the function to draw rectangles and save the annotated image
# draw_rectangles(json_data_path, image_path, output_image_path)
