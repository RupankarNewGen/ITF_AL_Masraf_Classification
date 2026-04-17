import os
import cv2
import json
from pathlib import Path

def validate_and_load_image(json_data, base_folder="Train"):
    """
    Validates and loads an image from the path specified in JSON data.
    
    Args:
        json_data (dict): JSON data containing image path
        base_folder (str): Base folder where images are stored
    
    Returns:
        tuple: (image array, status message)
    """
    try:
        # Extract image path from JSON
        image_path = json_data.get("meta", {}).get("image_path")
        if not image_path:
            return None, "No image path found in JSON data"

        # Construct full path
        abs_path = os.path.join(base_folder, image_path)
        # Convert to absolute path
        # abs_path = os.path.abspath(full_path)
        
        # Basic validation checks
        if not os.path.exists(abs_path):
            return None, f"Image file does not exist: {abs_path}"
            
        # if not os.access(abs_path, os.R_OK):
        #     return None, f"No read permissions for file: {abs_path}"
            
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        if not any(abs_path.lower().endswith(ext) for ext in valid_extensions):
            return None, f"Unsupported file format: {abs_path}"
            
        # Try to load the image
        image = cv2.imread(abs_path, 1)
        if image is None:
            return None, f"Failed to load image: {abs_path}"
            
        # Get expected dimensions from JSON
        expected_width = json_data.get("meta", {}).get("imageSize", {}).get("width")
        expected_height = json_data.get("meta", {}).get("imageSize", {}).get("height")
        
        # Verify image dimensions if provided
        if expected_width and expected_height:
            actual_height, actual_width = image.shape[:2]
            if actual_width != expected_width or actual_height != expected_height:
                print(f"Warning: Image dimensions mismatch. Expected: {expected_width}x{expected_height}, "
                      f"Got: {actual_width}x{actual_height}")
        
        return image, "Success"
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Example usage:
def process_dataset(json_file_path, base_folder):
    """
    Process a dataset containing multiple JSON entries
    """
    try:
        # Read JSON file
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
            
        image, message = validate_and_load_image(json_data, base_folder)
        
        if image is not None:
            print(f"Successfully loaded image with shape: {image.shape}")
            return image
        else:
            print(f"Failed to load image: {message}")
            return None
            
    except Exception as e:
        print(f"Error processing JSON file: {str(e)}")
        return None

# Example of how to use it:
if __name__ == "__main__":
    
    # img_path = '/home/gpu1admin/rakesh/ingram_rakesh_data/data_in_funsd_format/dataset/custom_geo/Test/Images/IM-000000010645000-AP_page_1_s_4.png'
    # image = cv2.imread(img_path, 1)
    # print('11111111122222222222222333333333333333')
    # print(image)
    # exit('OK')
    
    # Set your base folder where the Train folder is located
    base_folder = "/home/gpu1admin/rakesh/ingram_rakesh_data/data_in_funsd_format/dataset/custom_geo"
    # Read JSON file
    process_dataset_folder = '/home/gpu1admin/rakesh/ingram_rakesh_data/data_in_funsd_format/dataset/custom_geo/preprocessed'
    for json_file_path in os.listdir(process_dataset_folder):
        with open(os.path.join(process_dataset_folder, json_file_path), 'r') as f:
            json_data = json.load(f)
        # Process single image
        image, message = validate_and_load_image(json_data, base_folder)
        if image is not None:
            print(f"Image loaded successfully with shape: {image.shape}")
            # If you need to resize the image
            try:
                resized_image = cv2.resize(image, (800, 800))  # or whatever size you need
                print(f"Image resized successfully to shape: {resized_image.shape}")
            except Exception as e:
                raise SystemExit(f"Processing failed for json path: {json_file_path}")
                # print(f"Error resizing image: {str(e)}")
        else:
            print(f"Failed to load image: {message}")
            raise SystemExit(f"Processing failed for json path: {json_file_path}")
    print('PROCESS COMPLETED SUCCESSFULLY')