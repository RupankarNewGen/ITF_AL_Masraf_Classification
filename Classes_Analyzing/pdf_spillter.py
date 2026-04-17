from pdf2image import convert_from_path
import os

def pdf_to_images(pdf_path, output_folder):
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert PDF to list of images
    images = convert_from_path(pdf_path)

    # Save each page as JPEG
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        image.save(output_path, "JPEG")
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    pdf_path = "/home/lpt6964/Downloads/temp/Doc 5 ver 1.pdf"  # 🔁 replace with your PDF path
    output_folder = "/home/lpt6964/Downloads/temp/Doc_5"

    pdf_to_images(pdf_path, output_folder)