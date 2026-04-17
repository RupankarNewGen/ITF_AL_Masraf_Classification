# Importing Libraries
import numpy as np
import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pathlib import Path
import argparse
from datetime import datetime
import glob

# ANSI color codes for timing output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

language_mapper = {
        "english": "eng",
        "german": "deu",
        "french": "fra",
        "spanish": "spa",
        "hindi": "hin",
        "arabic": "ara"
    }


def check_negative_value(word_coordinates):
    has_negative_value = any(
        value < 0 if isinstance(value, (int, float)) else False for value in word_coordinates[-1].values())
    if has_negative_value:
        word_coordinates.pop()


def tesseract_generate_ocr_string_and_word_coordinates(image, ocr_confidence_threshold:int=0, language: str = "english"):
    stop_word = "false"
    word_dict_parent_path = "."
    blacklisted_character = ""
    psm = "6" ; oem = "3"
    language = language_mapper.get(language,"eng") #["arabic"]

    image = image.convert('RGB')
    np_array = np.array(image)
    image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)

    if stop_word == "true":
        word_dict_file_path = word_dict_parent_path + "/englist_stopwords.txt"
        custom_config = f"--psm {psm} --oem {oem} " \
                        f"-c tessedit_char_blacklist='{blacklisted_character}' " \
                        f"-c user_words={word_dict_file_path} -c load_system_dawg=false " \
                        f"-c load_freq_dawg=false"
    elif stop_word == "false":
        custom_config = f"--psm {psm} --oem {oem} " \
                        f"-c tessedit_char_blacklist='{blacklisted_character}'"

    word_coordinates = []

    # output the string only
    ocr_string:str = pytesseract.image_to_string(image, lang=language, config=custom_config)

    # output a dict from which we can get word conf
    ocr_data = pytesseract.image_to_data(image, lang=language, config=custom_config,
                                            output_type=pytesseract.Output.DICT)
    # print(ocr_data)
    ocr_confidence_based_string = ""
    for i, text in enumerate(ocr_data["text"]):
        if text != "":
            # Extract EXACT coordinates from Tesseract and store in multiple formats:
            # Format 1 (Original Tesseract format):
            #   - left: x-coordinate of top-left corner
            #   - top: y-coordinate of top-left corner
            #   - width: horizontal span of word
            #   - height: vertical span of word
            # Format 2 (Corner points):
            #   - x1, y1: top-left corner
            #   - x2, y2: bottom-right corner (calculated as left+width, top+height)
            # Format 3 (Bounding box array):
            #   - bbox: [left, top, right, bottom] for easy indexing
            word_coordinates.append({
                "word": text,
                "left": ocr_data["left"][i],
                "top": ocr_data["top"][i],
                "width": ocr_data["width"][i],
                "height": ocr_data["height"][i],
                "x1": ocr_data["left"][i],
                "y1": ocr_data["top"][i],
                "x2": ocr_data["left"][i] + ocr_data["width"][i],
                "y2": ocr_data["top"][i] + ocr_data["height"][i],
                "bbox": [ocr_data["left"][i], ocr_data["top"][i], 
                    ocr_data["left"][i] + ocr_data["width"][i], ocr_data["top"][i] + ocr_data["height"][i]],
                "confidence":ocr_data["conf"][i]
            })
            if ocr_data["conf"][i] > ocr_confidence_threshold:
                ocr_confidence_based_string += text
            check_negative_value(word_coordinates)
    
    if ocr_confidence_threshold > 0:
        return ocr_string, word_coordinates, ocr_confidence_based_string
    else:
        return ocr_string, word_coordinates


def visualize_bounding_boxes(image_path, word_coordinates, output_path=None):
    """
    Visualize bounding boxes of words on the image.
    
    BOUNDING BOX COORDINATE SYSTEM:
    ─────────────────────────────────
    Tesseract provides coordinates in pixel space:
    
    (x1, y1) ┌────────────────┐
    TOP-LEFT │                │ width
             │   WORD BOX     │
             │                │ height
             └────────────────┘ (x2, y2)
                             BOTTOM-RIGHT
    
    Where:
      • x1, y1 = top-left corner (tesseract's "left" and "top")
      • x2, y2 = bottom-right corner (calculated from left+width, top+height)
      • width = x2 - x1
      • height = y2 - y1
    
    This function draws rectangles using EXACT coordinates from Tesseract.
    
    Args:
        image_path (str): Path to the input image
        word_coordinates (list): List of word coordinate dictionaries
        output_path (str, optional): Path to save the visualization
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    print(f"{BLUE}Generating bounding box visualization...{RESET}")
    
    # Load image
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(image_rgb)
    
    # Draw bounding boxes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(word_coordinates)))
    
    for idx, word_data in enumerate(word_coordinates):
        # Extract exact coordinates from Tesseract
        x1 = word_data['x1']        # Left edge (from tesseract "left")
        y1 = word_data['y1']        # Top edge (from tesseract "top")
        x2 = word_data['x2']        # Right edge (left + width)
        y2 = word_data['y2']        # Bottom edge (top + height)
        word = word_data['word']
        confidence = word_data['confidence']
        
        # Draw rectangle using EXACT Tesseract coordinates
        # Rectangle constructor takes: (x_position, y_position), width, height
        rect = patches.Rectangle(
            (x1, y1),                    # Top-left corner (x, y)
            (x2 - x1),                   # Width of box
            (y2 - y1),                   # Height of box
            linewidth=1.5,
            edgecolor=colors[idx % len(colors)],
            facecolor='none'             # Transparent fill, only show outline
        )
        ax.add_patch(rect)
        
        # Add text label with confidence (reduced font size to minimize clutter)
        label = f"{word}" if confidence >= 0 else f"{word}"
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
    
    ax.set_title(f"Tesseract OCR Bounding Boxes - Total Words: {len(word_coordinates)}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"{GREEN}Visualization saved to: {output_path}{RESET}")
    
    return fig


def save_ocr_results(ocr_string, word_coordinates, image_path, output_folder=None, ocr_confidence_based_string=None):
    """
    Save OCR results to JSON and TXT files.
    
    Args:
        ocr_string (str): OCR extracted text
        word_coordinates (list): List of word coordinate dictionaries
        image_path (str): Path to the original image
        output_folder (str, optional): Output folder for results
        ocr_confidence_based_string (str, optional): OCR text filtered by confidence
    
    Returns:
        tuple: (json_output_path, txt_output_path)
    """
    if output_folder is None:
        output_folder = os.path.dirname(image_path)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate output filenames
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    json_output_path = os.path.join(output_folder, f"{image_basename}_textAndCoordinates.json")
    txt_output_path = os.path.join(output_folder, f"{image_basename}_textAndCoordinates.txt")
    
    # Prepare OCR data dictionary
    ocr_data = {
        'ocr_text': ocr_string,
        'word_coordinates': word_coordinates,
        'total_words': len(word_coordinates),
        'image_path': image_path,
        'timestamp': datetime.now().isoformat()
    }
    
    if ocr_confidence_based_string is not None:
        ocr_data['ocr_confidence_based_string'] = ocr_confidence_based_string
    
    # Save as JSON
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(ocr_data, f, indent=2, ensure_ascii=False)
    print(f"{GREEN}JSON output saved to: {json_output_path}{RESET}")
    
    # Save as TXT (human readable format)
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TESSERACT OCR RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Image: {image_path}\n")
        f.write(f"Total words detected: {len(word_coordinates)}\n")
        f.write(f"Timestamp: {ocr_data['timestamp']}\n\n")
        
        f.write("RAW TEXT:\n")
        f.write("-"*80 + "\n")
        f.write(ocr_string)
        f.write("\n\n")
        
        if ocr_confidence_based_string is not None:
            f.write("CONFIDENCE-FILTERED TEXT:\n")
            f.write("-"*80 + "\n")
            f.write(ocr_confidence_based_string)
            f.write("\n\n")
        
        f.write("WORD-BY-WORD WITH BOUNDING BOXES:\n")
        f.write("-"*80 + "\n")
        for idx, word_data in enumerate(word_coordinates, 1):
            f.write(f"{idx:4d}. Word: '{word_data['word']}'\n")
            f.write(f"     Position: ({word_data['x1']}, {word_data['y1']}) - ({word_data['x2']}, {word_data['y2']})\n")
            f.write(f"     Size: {word_data['width']} x {word_data['height']}\n")
            f.write(f"     Confidence: {word_data['confidence']}%\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("OCR DATA (STRUCTURED):\n")
        f.write("="*80 + "\n")
        f.write(str(ocr_data))
    
    print(f"{GREEN}TXT output saved to: {txt_output_path}{RESET}")
    
    return json_output_path, txt_output_path


def process_image(image_path, output_folder=None, language="english", 
                  ocr_confidence_threshold=0, show_visualization=True, 
                  save_visualization=True):
    """
    Main function to process an image with Tesseract OCR.
    
    Args:
        image_path (str): Path to the input image
        output_folder (str, optional): Output folder for results
        language (str): Language for OCR (default: english)
        ocr_confidence_threshold (int): Confidence threshold (default: 0)
        show_visualization (bool): Whether to display the visualization
        save_visualization (bool): Whether to save the visualization
    
    Returns:
        tuple: (ocr_string, word_coordinates, [ocr_confidence_based_string])
    """
    print(f"\n{'='*80}")
    print(f"TESSERACT OCR PROCESSING")
    print(f"{'='*80}\n")
    
    # Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"{BLUE}Processing: {image_path}{RESET}")
    print(f"Language: {language}, Confidence Threshold: {ocr_confidence_threshold}%\n")
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Extract OCR data
        start_time = datetime.now()
        if ocr_confidence_threshold > 0:
            ocr_string, word_coordinates, ocr_confidence_based_string = \
                tesseract_generate_ocr_string_and_word_coordinates(
                    image, ocr_confidence_threshold, language
                )
        else:
            ocr_string, word_coordinates = \
                tesseract_generate_ocr_string_and_word_coordinates(
                    image, ocr_confidence_threshold, language
                )
            ocr_confidence_based_string = None
        
        processing_time = datetime.now() - start_time
        print(f"{RED}OCR processing time: {processing_time}{RESET}\n")
        
        # Save OCR results
        if output_folder is None:
            output_folder = os.path.dirname(image_path) or "."
        
        json_path, txt_path = save_ocr_results(
            ocr_string, word_coordinates, image_path, 
            output_folder, ocr_confidence_based_string
        )
        
        # Generate visualization
        viz_output_path = None
        if save_visualization:
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            viz_output_path = os.path.join(output_folder, f"{image_basename}_bounding_boxes.png")
        
        fig = visualize_bounding_boxes(image_path, word_coordinates, viz_output_path)
        
        if show_visualization:
            plt.show()
        else:
            plt.close(fig)
        
        print(f"\n{GREEN}{'='*80}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved:")
        print(f"  - JSON: {json_path}")
        print(f"  - TXT: {txt_path}")
        if viz_output_path:
            print(f"  - Visualization: {viz_output_path}")
        print(f"  - Total words: {len(word_coordinates)}")
        print(f"{'='*80}{RESET}\n")
        
        if ocr_confidence_based_string is not None:
            return ocr_string, word_coordinates, ocr_confidence_based_string
        else:
            return ocr_string, word_coordinates
    
    except Exception as e:
        print(f"\n{RED}ERROR: {str(e)}{RESET}\n")
        raise


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


def process_folder(folder_path, output_folder=None, language="english", 
                   ocr_confidence_threshold=0, show_visualization=False, 
                   save_visualization=True):
    """
    Process all images in a folder.
    
    Args:
        folder_path (str): Path to the input folder
        output_folder (str, optional): Output folder for results
        language (str): Language for OCR
        ocr_confidence_threshold (int): Confidence threshold
        show_visualization (bool): Whether to display visualizations
        save_visualization (bool): Whether to save visualizations
    
    Returns:
        dict: Summary of processing results
    """
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING FOLDER")
    print(f"{'='*80}\n")
    
    # Validate folder path
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not found: {folder_path}")
    
    print(f"{BLUE}Processing folder: {folder_path}{RESET}")
    print(f"Language: {language}, Confidence Threshold: {ocr_confidence_threshold}%\n")
    
    # Set output folder
    if output_folder is None:
        output_folder = os.path.join(folder_path, "OCR_Results")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all images
    image_files = find_images_recursive(folder_path)
    
    if not image_files:
        print(f"{RED}No images found in {folder_path}{RESET}\n")
        return {'total': 0, 'processed': 0, 'failed': 0, 'images': []}
    
    print(f"{GREEN}Found {len(image_files)} images to process\n{RESET}")
    
    # Process each image
    results = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0,
        'images': [],
        'start_time': datetime.now().isoformat()
    }
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            
            # Load image
            image = Image.open(image_path)
            
            # Extract OCR data
            start_time = datetime.now()
            if ocr_confidence_threshold > 0:
                ocr_string, word_coordinates, ocr_confidence_based_string = \
                    tesseract_generate_ocr_string_and_word_coordinates(
                        image, ocr_confidence_threshold, language
                    )
            else:
                ocr_string, word_coordinates = \
                    tesseract_generate_ocr_string_and_word_coordinates(
                        image, ocr_confidence_threshold, language
                    )
                ocr_confidence_based_string = None
            
            processing_time = datetime.now() - start_time
            
            # Save OCR results
            json_path, txt_path = save_ocr_results(
                ocr_string, word_coordinates, image_path, 
                output_folder, ocr_confidence_based_string
            )
            
            # Generate visualization
            viz_output_path = None
            if save_visualization:
                image_basename = os.path.splitext(os.path.basename(image_path))[0]
                viz_output_path = os.path.join(output_folder, f"{image_basename}_bounding_boxes.png")
                fig = visualize_bounding_boxes(image_path, word_coordinates, viz_output_path)
                if not show_visualization:
                    plt.close(fig)
            
            results['images'].append({
                'image_path': image_path,
                'status': 'success',
                'words_detected': len(word_coordinates),
                'processing_time': str(processing_time),
                'json_output': json_path,
                'txt_output': txt_path,
                'viz_output': viz_output_path
            })
            results['processed'] += 1
            print(f"  ✅ Saved with {len(word_coordinates)} words\n")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}\n")
            results['images'].append({
                'image_path': image_path,
                'status': 'failed',
                'error': str(e)
            })
            results['failed'] += 1
    
    # Create summary report
    results['end_time'] = datetime.now().isoformat()
    summary_path = os.path.join(output_folder, "PROCESSING_SUMMARY.json")
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
    Command-line interface for OCR processing.
    """
    parser = argparse.ArgumentParser(
        description='Tesseract OCR with Bounding Box Visualization (Single Image or Batch Folder)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single image
  python tessaract.py /path/to/image.jpg
  python tessaract.py /path/to/image.jpg --output /output/folder
  
  # Batch process folder
  python tessaract.py /path/to/folder/
  python tessaract.py /path/to/folder/ --output /output/folder
  
  # With options
  python tessaract.py /path/to/image.jpg --language arabic --threshold 50
  python tessaract.py /path/to/image.jpg --no-show --no-save-viz
        '''
    )
    
    parser.add_argument(
        'path',
        help='Path to the input image file or folder (with trailing /)')
    
    
    parser.add_argument(
        '-o', '--output',
        dest='output_folder',
        help='Output folder for results (default: same as image folder)',
        default=None
    )
    
    parser.add_argument(
        '-l', '--language',
        dest='language',
        help='Language for OCR (english, german, french, spanish, hindi, arabic)',
        default='english',
        choices=['english', 'german', 'french', 'spanish', 'hindi', 'arabic']
    )
    
    parser.add_argument(
        '-t', '--threshold',
        dest='threshold',
        type=int,
        help='Confidence threshold for words (0-100, default: 0)',
        default=0
    )
    
    parser.add_argument(
        '--no-show',
        dest='show_visualization',
        action='store_false',
        default=True,
        help='Do not display visualization (default: show)'
    )
    
    parser.add_argument(
        '--no-save-viz',
        dest='save_visualization',
        action='store_false',
        default=True,
        help='Do not save visualization (default: save)'
    )
    
    args = parser.parse_args()
    
    # Determine if it's a folder or image
    if os.path.isdir(args.path):
        # Process folder
        process_folder(
            folder_path=args.path,
            output_folder=args.output_folder,
            language=args.language,
            ocr_confidence_threshold=args.threshold,
            show_visualization=args.show_visualization,
            save_visualization=args.save_visualization
        )
    else:
        # Process single image
        process_image(
            image_path=args.path,
            output_folder=args.output_folder,
            language=args.language,
            ocr_confidence_threshold=args.threshold,
            show_visualization=args.show_visualization,
            save_visualization=args.save_visualization
        )


if __name__ == "__main__":
    # Check if image path is provided via command line
    if len(__import__('sys').argv) > 1:
        main()
    else:
        # Default example usage if no arguments provided
        print(f"\n{BLUE}Running with example images...{RESET}")
        
        # Example 1
        try:
            example_image_1 = "./blank_page_samples/blank_image_1.png"
            if os.path.exists(example_image_1):
                process_image(example_image_1, show_visualization=False)
        except Exception as e:
            print(f"{RED}Example 1 failed: {e}{RESET}")
        
        # Example 2
        try:
            example_image_2 = "./blank_page_samples/textual_image.png"
            if os.path.exists(example_image_2):
                process_image(example_image_2, show_visualization=False)
        except Exception as e:
            print(f"{RED}Example 2 failed: {e}{RESET}")
        
        print(f"\n{GREEN}Usage: python {__import__('sys').argv[0]} <image_path_or_folder> [options]{RESET}")
        print(f"{GREEN}       Run with --help for more options{RESET}")
        print(f"{GREEN}       For folder input, add trailing slash: /path/to/folder/{RESET}\n")
