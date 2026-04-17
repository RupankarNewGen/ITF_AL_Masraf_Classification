import os

def filter_nonexistent_images(input_txt, images_folder, output_txt):
    # Read the file names from the input text file
    with open(input_txt, 'r') as file:
        image_names = file.read().splitlines()
    
    # Get the list of files in the images folder
    folder_images = set(os.listdir(images_folder))
    
    # Filter the image names that do not exist in the folder 
    missing_images = [img for img in  folder_images if img not in image_names]
    
    # Write the missing image names to the output text file
    with open(output_txt, 'w') as file:
        file.write('\n'.join(missing_images))
    
    print(f"Filtered {len(missing_images)} missing images. Results saved to {output_txt}.")

# Example usage
input_txt = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Train_data/CI/CI_verified_annotations/train_data/original_train_data/test.txt'  # Replace with the path to your input text file
images_folder = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Train_data/CI/CI_verified_annotations/train_data/original_train_data/Images'       # Replace with the path to your images folder
output_txt = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Train_data/CI/CI_verified_annotations/train_data/original_train_data/train.txt'  # Output text file for missing images

filter_nonexistent_images(input_txt, images_folder, output_txt)