import os
import shutil

def collect_matching_ocr(image_root, ocr_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    # Step 1: Collect all expected OCR filenames from images
    expected_ocr_files = set()

    for root, _, files in os.walk(image_root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                base = os.path.splitext(file)[0]
                ocr_name = base + "_textAndCoordinates.txt"
                expected_ocr_files.add(ocr_name)

    print(f"Expected OCR files: {len(expected_ocr_files)}")

    # Step 2: Find and copy matching OCR files
    copied = 0

    for root, _, files in os.walk(ocr_root):
        for file in files:
            if file in expected_ocr_files:
                src = os.path.join(root, file)
                dst = os.path.join(output_root, file)

                shutil.copy2(src, dst)
                copied += 1

    print(f"Copied OCR files: {copied}")


# 🔹 Usage
image_folder = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Geo_layout_codebase/Geo_LayoutLM-geo_code_latest_timeoptm_version_updated_apr22_sharing_backend/training_main/main/CS_Data/CS_annoation/images"
ocr_folder = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/ocr_data"
output_folder = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Geo_layout_codebase/Geo_LayoutLM-geo_code_latest_timeoptm_version_updated_apr22_sharing_backend/training_main/main/CS_Data/CS_annoation/OCR"

collect_matching_ocr(image_folder, ocr_folder, output_folder)