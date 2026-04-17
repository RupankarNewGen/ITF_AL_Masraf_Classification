import os
import base64
from pdf2image import convert_from_path

def pdf_to_png_and_base64(pdf_path, output_dir, dpi=300):
    os.makedirs(output_dir, exist_ok=True)

    pages = convert_from_path(pdf_path, dpi=dpi)

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for i, page in enumerate(pages):
        page_num = i + 1

        # PNG path
        png_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num}.png")
        page.save(png_path, format="PNG")

        # Read image and convert to base64
        with open(png_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")

        # Base64 txt path
        txt_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num}.txt")
        with open(txt_path, "w") as txt_file:
            txt_file.write(encoded_string)

        print(f"Saved: {png_path} and {txt_path}")

# Example usage
pdf_path = "/home/lpt6964/Downloads/ITF_utils/ITF_AL_MASRAF_Documents/Import Bills/mohammedkm_23-01-2026_8-50-01_23.pdf"
output_dir = "/home/lpt6964/Downloads/ITF_utils/AL_Masraf_23"

pdf_to_png_and_base64(pdf_path, output_dir)
