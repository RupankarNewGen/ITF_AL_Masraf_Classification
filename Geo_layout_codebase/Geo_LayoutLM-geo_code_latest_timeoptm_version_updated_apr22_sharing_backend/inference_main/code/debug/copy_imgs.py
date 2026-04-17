
import os
import shutil
from fuzzywuzzy import fuzz



def copy_images_to_folder(image_list, destination_folder,  root_data_img, root_data_anno):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    destination_imag_root = os.path.join(destination_folder, 'Images')
    destination_anno_root = os.path.join(destination_folder, 'Labels')
    
    # imag_root = os.path.join(root_data, 'Images')
    # anno_root = os.path.join(root_data, 'Labels')
    
    # Loop through each image in the list
    for image_path in image_list:
        # Check if the image file exists
        if os.path.isfile(os.path.join(root_data_img, image_path)):
            # Copy the image to the destination folder
            shutil.copy(os.path.join(root_data_img, image_path), destination_imag_root)
            image_path = image_path.replace('.png', '.txt')
            shutil.copy(os.path.join(root_data_anno, image_path), destination_anno_root)
            print(f"Copied: {image_path}")
        else:
            print(f"File not found: {image_path}")


def ocr_file_copy(image_list, destination_folder_ocr):
    
    for image_path in image_list:
        # Check if the image file exists
        if os.path.isfile(os.path.join(root_data_ocr, image_path)):
            # Copy the image to the destination folder
            shutil.copy(os.path.join(root_data_ocr, image_path), destination_folder_ocr)
            print(f"Copied: {image_path}")
        else:
            print(f"File not found: {image_path}")
from fuzzywuzzy import fuzz

# Example usage:
if __name__ == '__main__':
    def clean_text(text):
        # Remove leading and trailing '/', '-', or ':'
        token_to_strip = "/-:"
        return text.strip(token_to_strip)
	
    # Example texts
    pred = "/ 54839"
    actual = "54839"

    # Cleaned results
    cleaned_pred = clean_text(pred)
    cleaned_actual = clean_text(actual)

    print("Cleaned Pred:", cleaned_pred)
    print("Cleaned Actual:", cleaned_actual)
    print(fuzz.ratio(cleaned_pred.replace(" ", ""), cleaned_actual.replace(" ", ""))) # fuzz.partial_ratio(value1, value2)
    print(fuzz.ratio(cleaned_pred, cleaned_actual)) # fuzz.partial_ratio(value1, value2)

    exit('OK')
    def get_class_names(dataset_root_path):
        class_names_file = os.path.join(dataset_root_path)#, "class_names.txt")
        class_names = (
            open(class_names_file, "r", encoding="utf-8").read().strip().split("\n")
        )
        return class_names
    
    
    class_names = get_class_names('/media/ntlpt19/5250315B5031474F/ingram_rakesh_data/class_names.txt')
    # print(f'class names:{self.class_names}')
    class_idx_dic = dict(
        [(class_name, idx) for idx, class_name in enumerate(class_names)]
    )
    print(f'classes in dict: {class_idx_dic}')
    exit('111111111  +++++++++++++++++')
    
    
    
    
    
    
    
    
    
    
    
    
    
    split_file = '/media/ntlpt19/5250315B5031474F/ingram_rakesh_data/test.txt'
    root_data_img = '/media/ntlpt19/5250315B5031474F/ingram_rakesh_data/images'
    root_data_anno = '/media/ntlpt19/5250315B5031474F/ingram_rakesh_data/labels'
    root_data_ocr = '/media/ntlpt19/5250315B5031474F/ingram_rakesh_data/OCR'

    with open(split_file, 'r') as file:
        img_files_list = [line.strip() for line in file.readlines()]
        
    # annotation_txt = [filename.replace('.png', '.txt') for filename in img_files_list]
    ocr_files = [filename.replace('.png', '_textAndCoordinates.txt') for filename in img_files_list]

    destination_folder = "/media/ntlpt19/5250315B5031474F/ingram_rakesh_data/eval_data"
    destination_folder_ocr = "/media/ntlpt19/5250315B5031474F/ingram_rakesh_data/eval_data/OCR"

    # copy_images_to_folder(img_files_list, destination_folder, root_data_img, root_data_anno)
    ocr_file_copy(ocr_files, destination_folder_ocr)