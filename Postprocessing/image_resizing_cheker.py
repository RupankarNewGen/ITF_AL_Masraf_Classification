import os
import io
from PIL import Image
from tqdm import tqdm

def replicate_exact_processing(source_folder, output_folder, target_dpi=None, target_size=None):
    """
    Exactly replicates the previous logic:
    1. Rescales by DPI if target_dpi is provided.
    2. Forces the image into target_size dimensions (may stretch/squash).
    3. Saves as JPEG.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    image_files = [f for f in os.listdir(source_folder) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"Processing {len(image_files)} images using EXACT previous logic.")

    for filename in tqdm(image_files, desc="Resizing"):
        img_path = os.path.join(source_folder, filename)
        save_path = os.path.join(output_folder, filename)

        try:
            with Image.open(img_path) as img:
                # --- Step 1: DPI Scaling ---
                if target_dpi is not None:
                    current_dpi = img.info.get('dpi', (300, 300))[0]
                    scale_factor = target_dpi / current_dpi
                    if scale_factor != 1.0:
                        new_w = int(img.width * scale_factor)
                        new_h = int(img.height * scale_factor)
                        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # --- Step 2: Exact Resize (Forced Dimensions) ---
                if target_size is not None:
                    # This replicates the exact (1152, 896) forcing from before
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # --- Step 3: Save as JPEG ---
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                img.save(save_path, format="JPEG", quality=95)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Set your paths here
    INPUT = "/datadrive2/IDF_AL_MASRAF/test_images"
    OUTPUT = "/datadrive2/IDF_AL_MASRAF/test_images_resized"
    
    # Matching your previous API settings exactly
    T_SIZE = (896,1152)  # Note the order (width, height) for PIL
    T_DPI = None 
    
    replicate_exact_processing(INPUT, OUTPUT, target_dpi=T_DPI, target_size=T_SIZE)