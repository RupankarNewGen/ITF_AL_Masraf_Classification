import os
import argparse


def delete_images_without_ocr(images_dir, labels_dir, ocr_dir, image_extensions=None):
    """
    Deletes images and corresponding label files if the OCR file is not present.

    Args:
        images_dir (str): Path to the images directory.
        labels_dir (str): Path to the labels directory.
        ocr_dir (str): Path to the OCR directory.
        image_extensions (tuple): Supported image file extensions.
    """
    if image_extensions is None:
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    deleted_images = 0
    deleted_labels = 0
    missing_ocr_count = 0

    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(image_extensions):
            continue

        base_name = os.path.splitext(filename)[0]

        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        ocr_path = os.path.join(ocr_dir, f"{base_name}_textAndCoordinates.txt")

        # Check if OCR file exists
        if not os.path.exists(ocr_path):
            missing_ocr_count += 1
            print(f"OCR missing for: {base_name}")

            # Delete image
            if os.path.exists(image_path):
                os.remove(image_path)
                deleted_images += 1
                print(f"Deleted image: {image_path}")

            # Delete corresponding label
            if os.path.exists(label_path):
                os.remove(label_path)
                deleted_labels += 1
                print(f"Deleted label: {label_path}")

    print("\nSummary:")
    print(f"Total images with missing OCR: {missing_ocr_count}")
    print(f"Total images deleted: {deleted_images}")
    print(f"Total labels deleted: {deleted_labels}")


def main():
    parser = argparse.ArgumentParser(
        description="Delete images and labels for which OCR files are missing."
    )
    parser.add_argument(
        "--images_dir", required=True, help="Path to the images directory"
    )
    parser.add_argument(
        "--labels_dir", required=True, help="Path to the labels directory"
    )
    parser.add_argument(
        "--ocr_dir", required=True, help="Path to the OCR directory"
    )

    args = parser.parse_args()

    delete_images_without_ocr(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        ocr_dir=args.ocr_dir,
    )


if __name__ == "__main__":
    main()