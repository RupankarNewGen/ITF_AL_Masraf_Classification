import os
from PIL import Image

def convert_tiff_to_jpeg(input_tiff_path, output_base_dir, quality=95):
    # Get TIFF filename without extension
    tiff_name = os.path.splitext(os.path.basename(input_tiff_path))[0]
    
    # Create output subfolder
    output_dir = os.path.join(output_base_dir, tiff_name)
    os.makedirs(output_dir, exist_ok=True)

    with Image.open(input_tiff_path) as img:
        page = 0

        while True:
            try:
                img.seek(page)
                
                # Convert to RGB (JPEG does not support RGBA/CMYK properly)
                rgb_img = img.convert("RGB")
                
                output_path = os.path.join(
                    output_dir,
                    f"{tiff_name}_page_{page+1}.jpg"
                )
                
                # Save as JPEG at 300 DPI
                rgb_img.save(
                    output_path,
                    "JPEG",
                    dpi=(300, 300),
                    quality=quality
                )

                print(f"Saved: {output_path}")
                page += 1

            except EOFError:
                break

    print("Conversion completed.")


# Example usage
if __name__ == "__main__":
    input_tiff = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/AlmasRaf_test_pdfs/Trade Finance_20210816122339_1.00.TIF"
    output_directory = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/AlmasRaf_test_pdfs/jepg_files"
    
    convert_tiff_to_jpeg(input_tiff, output_directory)