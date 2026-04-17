import os
import ast
from pathlib import Path

# ------------------- Directory Configuration -------------------
# Where your subfolder-structured images are stored
IMAGE_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Raw_ouput_data/FINAL_merged_classified_images"

# The flat folder containing all the ABBYY textAndCoordinates.txt files
ABBYY_FLAT_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/abby_clean_ocr"

# Where you want the mirrored _alltext.txt files saved
OUTPUT_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/abby_clean_ocr"

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")

def extract_text_from_abbyy(filepath):
    """Parses the ABBYY dictionary format and extracts all words into a single string."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Safely evaluate the string as a Python dictionary
        data = ast.literal_eval(content)
        
        # Extract the 'word' value from each dictionary in the 'ocrContent' list
        words = [item['word'] for item in data.get('ocrContent', []) if 'word' in item]
        
        # Join words with a space
        return " ".join(words)
        
    except SyntaxError:
        print(f"❌ Syntax error parsing file: {filepath}. Is it formatted correctly?")
        return ""
    except Exception as e:
        print(f"❌ Error reading {filepath}: {str(e)}")
        return ""

def main():
    print(f"Scanning images in: {IMAGE_DIR}")
    print(f"Looking for ABBYY files in: {ABBYY_FLAT_DIR}")
    
    success_count = 0
    missing_count = 0

    # 1. Traverse the nested image directory
    for root, dirs, files in os.walk(IMAGE_DIR):
        for filename in files:
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue

            # 2. Derive the base name
            base_name = Path(filename).stem
            
            # 3. Create a list of possible ABBYY filenames to look for
            possible_names = [
                f"{base_name}_textAndCoordinates.txt",                 # Exact match
                f"{base_name.replace(' ', '_')}_textAndCoordinates.txt", # Spaces replaced with underscores
                f"{filename}_textAndCoordinates.txt"                   # Extension included (.jpeg_text...)
            ]

            abbyy_filepath = None
            
            # Check which variation actually exists in the folder
            for name in possible_names:
                temp_path = os.path.join(ABBYY_FLAT_DIR, name)
                if os.path.exists(temp_path):
                    abbyy_filepath = temp_path
                    break

            # 4. If none of the variations exist, log it and skip
            if not abbyy_filepath:
                print(f"⚠️ Missing OCR for image: {filename}")
                missing_count += 1
                continue

            # 5. Extract the continuous text
            extracted_text = extract_text_from_abbyy(abbyy_filepath)

            # 6. Calculate relative path to mirror the directory structure
            rel_path = os.path.relpath(root, IMAGE_DIR)
            target_dir = os.path.join(OUTPUT_DIR, rel_path)
            os.makedirs(target_dir, exist_ok=True)

            # 7. Save the _alltext.txt file
            output_filename = os.path.join(target_dir, f"{base_name}_alltext.txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(extracted_text)
                
            success_count += 1
            print(f"✅ Saved: {os.path.join(rel_path, f'{base_name}_alltext.txt')}")

    print("\n--- Summary ---")
    print(f"Successfully formatted: {success_count} files.")
    print(f"Missing OCR files: {missing_count} files.")

if __name__ == "__main__":
    main()