import os
import shutil
import time
import moment
from imagededup.methods import PHash

def remove_duplicate_images(
    image_directory: str,
    unique_docs_dir: str,
    unique_duplicates_images_directory: str,
    log_dir: str
):

    phasher = PHash()
    
    # Encode images
    encodings = phasher.encode_images(image_dir=image_directory)
 
    # Find duplicate clusters
    duplicates = phasher.find_duplicates(encoding_map=encodings,max_distance_threshold=15)
 
    visited = set()
    layout_id = 0
    all_images = set(os.listdir(image_directory))
 
    # Create layout folders
    for key, dup_list in duplicates.items():
        if key in visited:
            continue
 
        group = set([key] + dup_list)
 
        # Only treat as layout if group size > 1
        if len(group) > 1:
            layout_id += 1
            layout_path = os.path.join(
                unique_duplicates_images_directory,
                f"layout_{layout_id}"
            )

            os.makedirs(layout_path, exist_ok=True)
 
            for img in group:
                src = os.path.join(image_directory, img)
                shutil.copy(src, os.path.join(layout_path, img))
                shutil.copy(src, os.path.join(unique_docs_dir, img))
 
            visited.update(group)
 
    # Images with no similar layout
    others = all_images - visited
    others_path = os.path.join(unique_duplicates_images_directory, "others")
    os.makedirs(others_path, exist_ok=True)
 
    for img in others:
        shutil.copy(
            os.path.join(image_directory, img),
            os.path.join(others_path, img)
        )
 
    print("========== SUMMARY ==========")
    print(f"Total images            : {len(all_images)}")
    print(f"Total layouts created   : {layout_id}")
    print(f"Images in layouts       : {len(visited)}")
    print(f"Images saved in others  : {len(others)}")
    print("================================")
 
if __name__ == "__main__": 
    image_directory = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/FINAL_merged_classified_images/Bill_of_Exchange"
    output_path = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/FINAL_layouts_of_merged_classified_images/BILL_of_exchange2"
    unique_duplicates_images_directory = os.path.join(output_path, "duplicates_removal_res")
    unique_docs_dir = os.path.join(output_path, "unique_docs")
    log_dir = os.path.join(output_path, "logs")
    os.makedirs(unique_duplicates_images_directory, exist_ok=True)
    os.makedirs(unique_docs_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
 
    todays_date = moment.unix(time.time(), utc=True).locale("Asia/Kolkata").format("YYYY-MM-DD_HH-mm-ss")
 
    remove_duplicate_images(
        image_directory=image_directory,
        unique_docs_dir=unique_docs_dir,
        unique_duplicates_images_directory=unique_duplicates_images_directory,
        log_dir=log_dir
    )
 