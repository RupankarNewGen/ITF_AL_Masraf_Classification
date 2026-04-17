import cv2

def draw_bbox(image_path, bbox, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Image not found or cannot be read.")
        return

    # Bounding box coordinates: [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = bbox

    # Draw the bounding box on the image
    color = (0, 0, 255)  # Green color for the box
    thickness = 5        # Line thickness
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    # Save the modified image
    cv2.imwrite(output_path, image)
    print(f"Image with bounding box saved to {output_path}")

# Inputs
image_path = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/CI/v2/Results_with_aug/Invoice_41_16.png"
output_path = "output_image.jpg"       # Replace with desired output path
bbox = [146, 1026, 561, 1049]          # Bounding box coordinates

# Draw and save the bounding box
draw_bbox(image_path, bbox, output_path)
exit('>>>>>>>>>>>>>>>>>')










import cv2

# Load the image
image_path = "/home/ntlpt19/Desktop/TF_release/Geo_LayoutLM/inference_main/code/output_image.jpg"
output_path = "output_image.jpg"  # Path to save the output image
image = cv2.imread(image_path)

bounding_box = {'word': 'oa', 'text': 'oa', 'confidence': 23, 'vertices': [[1564, 1829], [1683, 1829], [1683, 1852], [1564, 1852]], 'left': 1564, 'top': 1829, 'width': 119, 'height': 23, 'x1': 1564, 'y1': 1829, 'x2': 1683, 'y2': 1852}

# Extract bounding box details
bounding_box1 = {
    'word': 'nen',
    'text': 'nen',
    'confidence': 56,
    'vertices': [[1460, 1829], [1516, 1829], [1516, 1852], [1460, 1852]],
    'left': 1460,
    'top': 1829,
    'width': 56,
    'height': 23,
    'x1': 1460,
    'y1': 1829,
    'x2': 1516,
    'y2': 1852
}

# Get coordinates for the bounding box
x1, y1 = bounding_box['x1'], bounding_box['y1']
x2, y2 = bounding_box['x2'], bounding_box['y2']

# Draw the rectangle on the image
color = (0, 255, 0)  # Green color for the bounding box
thickness = 2
cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# Add the text inside or near the bounding box
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (0, 0, 255)  # Red color for the text
line_type = 2
text_position = (x1, y1 - 10)  # Adjust position slightly above the box
cv2.putText(image, bounding_box['text'], text_position, font, font_scale, font_color, line_type)

# Save the output image
cv2.imwrite(output_path, image)
print(f"Output image saved to {output_path}")
