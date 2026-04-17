# Importing Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pathlib import Path
import argparse
from datetime import datetime
import glob
import ast
import re

# ANSI color codes for timing output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def parse_ocr_file(ocr_file_path):
    """
    Parse OCR data from text file.
    
    The OCR files contain a Python dictionary representation as text.
    This function reads and parses the dictionary safely.
    
    Args:
        ocr_file_path (str): Path to the OCR text file
    
    Returns:
        dict: Parsed OCR data with 'ocrContent' key
    """
    try:
        with open(ocr_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Parse the dictionary string
        ocr_data = ast.literal_eval(content)
        
        if 'ocrContent' not in ocr_data:
            raise ValueError("OCR file does not contain 'ocrContent' key")
        
        return ocr_data
    
    except Exception as e:
        raise ValueError(f"Failed to parse OCR file {ocr_file_path}: {str(e)}")


def find_matching_ocr_file(image_path, ocr_folder):
    """
    Find the corresponding OCR file for an image.
    
    Matching logic:
    - Extract image basename: A09862_20210125163753_1.00_page_1.jpg
    - Remove extension: A09862_20210125163753_1.00_page_1
    - Look for: A09862_20210125163753_1.00_page_1_textAndCoordinates.txt
    
    Args:
        image_path (str): Path to the image file
        ocr_folder (str): Folder containing OCR files
    
    Returns:
        str: Path to matching OCR file, or None if not found
    """
    image_basename = os.path.basename(image_path)
    image_name = os.path.splitext(image_basename)[0]  # Remove extension
    
    # OCR file naming convention: {image_name}_textAndCoordinates.txt
    ocr_filename = f"{image_name}_textAndCoordinates.txt"
    ocr_path = os.path.join(ocr_folder, ocr_filename)
    
    if os.path.exists(ocr_path):
        return ocr_path
    
    # Alternative: try with JSON extension
    ocr_filename_json = f"{image_name}_textAndCoordinates.json"
    ocr_path_json = os.path.join(ocr_folder, ocr_filename_json)
    
    if os.path.exists(ocr_path_json):
        with open(ocr_path_json, 'r', encoding='utf-8') as f:
            return ocr_path_json
    
    return None


def load_image_with_cv2(image_path):
    """
    Load image using OpenCV and convert to RGB.
    
    Args:
        image_path (str): Path to the image
    
    Returns:
        numpy.ndarray: Image in RGB format
    """
    import cv2
    import numpy as np
    
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return image_rgb


def visualize_ocr_data(image_path, ocr_data, output_path=None):
    """
    Visualize bounding boxes from OCR data on the image.
    Same as tessract.py visualization but reading from existing OCR files.
    
    BOUNDING BOX COORDINATE SYSTEM:
    ─────────────────────────────────
    OCR coordinates in pixel space:
    
    (x1, y1) ┌────────────────┐
    TOP-LEFT │                │ width
             │   WORD BOX     │
             │                │ height
             └────────────────┘ (x2, y2)
                             BOTTOM-RIGHT
    
    Args:
        image_path (str): Path to the image
        ocr_data (dict): OCR data with 'ocrContent' containing word boxes
        output_path (str, optional): Path to save visualization
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    print(f"{BLUE}Generating OCR bounding box visualization...{RESET}")
    
    # Load image
    image_rgb = load_image_with_cv2(image_path)
    
    # Get word coordinates
    word_coordinates = ocr_data.get('ocrContent', [])
    
    if not word_coordinates:
        print(f"{YELLOW}Warning: No OCR content found in data{RESET}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(image_rgb)
    
    # Draw bounding boxes
    import numpy as np
    colors = plt.cm.rainbow(np.linspace(0, 1, len(word_coordinates)))
    
    for idx, word_data in enumerate(word_coordinates):
        # Extract exact coordinates from OCR data
        # OCR files can have multiple coordinate formats; try them in order
        if 'x1' in word_data and 'y1' in word_data and 'x2' in word_data and 'y2' in word_data:
            x1 = word_data['x1']        # Left edge
            y1 = word_data['y1']        # Top edge
            x2 = word_data['x2']        # Right edge
            y2 = word_data['y2']        # Bottom edge
        elif 'bbox' in word_data:
            x1, y1, x2, y2 = word_data['bbox']
        elif 'left' in word_data and 'top' in word_data:
            x1 = word_data['left']
            y1 = word_data['top']
            x2 = x1 + word_data.get('width', 0)
            y2 = y1 + word_data.get('height', 0)
        else:
            continue  # Skip if no valid coordinates
        
        word = word_data.get('word', '')
        confidence = word_data.get('confidence', -1)
        
        # Draw rectangle using EXACT OCR coordinates
        rect = patches.Rectangle(
            (x1, y1),                    # Top-left corner (x, y)
            (x2 - x1),                   # Width of box
            (y2 - y1),                   # Height of box
            linewidth=1.5,
            edgecolor=colors[idx % len(colors)],
            facecolor='none'             # Transparent fill, only outline
        )
        ax.add_patch(rect)
        
        # Add text label (reduced font size to minimize clutter)
        label = f"{word}"
        ax.text(
            x1,
            y1 - 3,
            label,
            fontsize=4,
            color=colors[idx % len(colors)],
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5),
            ha='left',
            va='bottom'
        )
    
    ax.set_title(f"OCR Data Visualization - Total Words: {len(word_coordinates)}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"{GREEN}Visualization saved to: {output_path}{RESET}")
    
    return fig


def find_images_recursive(folder_path):
    """
    Recursively find all image files in a folder.
    Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
    """
    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                       '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF')
    
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    
    return sorted(image_files)


def process_image_with_ocr(image_path, ocr_folder, output_folder):
    """
    Process a single image with its corresponding OCR data.
    
    Args:
        image_path (str): Path to the image
        ocr_folder (str): Folder containing OCR files
        output_folder (str): Output folder for visualizations
    
    Returns:
        dict: Processing result
    """
    try:
        image_basename = os.path.basename(image_path)
        print(f"{BLUE}Processing: {image_basename}{RESET}")
        
        # Find matching OCR file
        ocr_file_path = find_matching_ocr_file(image_path, ocr_folder)
        
        if ocr_file_path is None:
            return {
                'image': image_basename,
                'status': 'failed',
                'error': 'No matching OCR file found'
            }
        
        print(f"  Found OCR file: {os.path.basename(ocr_file_path)}")
        
        # Parse OCR data
        ocr_data = parse_ocr_file(ocr_file_path)
        num_words = len(ocr_data.get('ocrContent', []))
        
        # Generate visualization
        image_name = os.path.splitext(image_basename)[0]
        viz_output_path = os.path.join(output_folder, f"{image_name}_ocr_visualization.png")
        
        fig = visualize_ocr_data(image_path, ocr_data, viz_output_path)
        plt.close(fig)
        
        print(f"  ✅ Visualized {num_words} words\n")
        
        return {
            'image': image_basename,
            'status': 'success',
            'ocr_file': os.path.basename(ocr_file_path),
            'words_detected': num_words,
            'visualization': viz_output_path
        }
    
    except Exception as e:
        print(f"  ❌ Failed: {str(e)}\n")
        return {
            'image': image_basename,
            'status': 'failed',
            'error': str(e)
        }


def process_images_folder(images_folder, ocr_folder, output_folder):
    """
    Process all images in a folder, matching with existing OCR data.
    
    Args:
        images_folder (str): Folder containing images
        ocr_folder (str): Folder containing OCR files
        output_folder (str): Output folder for visualizations
    
    Returns:
        dict: Summary of processing results
    """
    print(f"\n{'='*80}")
    print(f"OCR DATA VISUALIZATION - BATCH PROCESSING")
    print(f"{'='*80}\n")
    
    # Validate folders
    if not os.path.isdir(images_folder):
        raise NotADirectoryError(f"Images folder not found: {images_folder}")
    if not os.path.isdir(ocr_folder):
        raise NotADirectoryError(f"OCR folder not found: {ocr_folder}")
    
    print(f"{BLUE}Images folder: {images_folder}{RESET}")
    print(f"{BLUE}OCR folder: {ocr_folder}{RESET}")
    print(f"{BLUE}Output folder: {output_folder}{RESET}\n")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all images
    image_files = find_images_recursive(images_folder)
    
    if not image_files:
        print(f"{RED}No images found in {images_folder}{RESET}\n")
        return {'total': 0, 'processed': 0, 'failed': 0, 'results': []}
    
    print(f"{GREEN}Found {len(image_files)} images to process\n{RESET}")
    
    # Process each image
    results = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0,
        'results': [],
        'start_time': datetime.now().isoformat()
    }
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] ", end="")
        
        result = process_image_with_ocr(image_path, ocr_folder, output_folder)
        results['results'].append(result)
        
        if result['status'] == 'success':
            results['processed'] += 1
        else:
            results['failed'] += 1
    
    # Create summary report
    results['end_time'] = datetime.now().isoformat()
    summary_path = os.path.join(output_folder, "VISUALIZATION_SUMMARY.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{GREEN}{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_folder}")
    print(f"\nSummary:")
    print(f"  Total images: {results['total']}")
    print(f"  Successfully processed: {results['processed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Summary report: {summary_path}")
    print(f"{'='*80}{RESET}\n")
    
    return results


def main():
    """
    Command-line interface for OCR data visualization.
    """
    parser = argparse.ArgumentParser(
        description='Visualize existing OCR data with bounding boxes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Visualize all images in a folder using their OCR data
  python visualize_ocr_data.py /path/to/images/ /path/to/ocr_data/ /path/to/output/
  
  # Short form
  python visualize_ocr_data.py /images/ /ocr_data/ /visualizations/
        '''
    )
    
    parser.add_argument(
        'images_folder',
        help='Path to folder containing images'
    )
    
    parser.add_argument(
        'ocr_folder',
        help='Path to folder containing OCR data files'
    )
    
    parser.add_argument(
        'output_folder',
        help='Path to output folder for visualizations'
    )
    
    args = parser.parse_args()
    
    # Process images folder
    process_images_folder(
        images_folder=args.images_folder,
        ocr_folder=args.ocr_folder,
        output_folder=args.output_folder
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        print(f"\n{BLUE}OCR Data Visualization Script{RESET}")
        print(f"{GREEN}Usage: python visualize_ocr_data.py <images_folder> <ocr_folder> <output_folder>{RESET}")
        print(f"{GREEN}       Run with --help for more options{RESET}\n")
