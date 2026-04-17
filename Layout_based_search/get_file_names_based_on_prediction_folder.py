import os
import json
from pathlib import Path
from collections import defaultdict

def filter_images_by_classification(results_folder, target_classes=None, 
                                    output_file=None, exclude_classes=None):
    """
    Filter image names from model prediction JSON files by classification.
    
    Args:
        results_folder: Folder containing model output JSON files
        target_classes: List of classes to filter for (e.g., ['Bill_of_Exchange', 'Commercial_Invoice'])
                       If None, returns all classes found
        output_file: Optional path to save filtered results as JSON
        exclude_classes: List of classes to exclude (e.g., ['others', 'error'])
    
    Returns:
        Dictionary with structure:
        {
            'class_name': {
                'count': int,
                'images': ['image1.jpg', 'image2.jpg', ...]
            },
            'summary': {...}
        }
    """
    
    if not os.path.isdir(results_folder):
        raise ValueError(f"Results folder not found: {results_folder}")
    
    if target_classes is None:
        target_classes = []
    
    if exclude_classes is None:
        exclude_classes = []
    
    # Normalize class names
    target_classes = [c.lower() for c in target_classes]
    exclude_classes = [c.lower() for c in exclude_classes]
    
    results = defaultdict(lambda: {'count': 0, 'images': []})
    all_files_count = 0
    error_count = 0
    
    print(f"\n{'='*80}")
    print(f"IMAGE CLASSIFICATION FILTER")
    print(f"{'='*80}\n")
    
    print(f"📁 Results Folder: {results_folder}")
    print(f"🎯 Target Classes: {target_classes if target_classes else 'All'}")
    print(f"❌ Exclude Classes: {exclude_classes if exclude_classes else 'None'}")
    print(f"💾 Output File: {output_file if output_file else 'None (console only)'}")
    print()
    
    # Find all JSON files
    json_files = [f for f in os.listdir(results_folder) if f.endswith('.json')]
    
    print(f"🔍 Found {len(json_files)} JSON files\n")
    print(f"{'─'*80}\n")
    
    # Process each JSON file
    for json_file in json_files:
        all_files_count += 1
        json_path = os.path.join(results_folder, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract classification
            classification = data.get('classification', 'unknown')
            classification_lower = classification.lower()
            
            # Extract image name (remove _classification.json suffix if present)
            image_name = json_file.replace('_classification.json', '').replace('.json', '')
            
            # Check if should be included
            should_include = True
            
            # If target classes specified, must match one of them
            if target_classes:
                should_include = classification_lower in target_classes
            
            # If in exclude list, skip it
            if classification_lower in exclude_classes:
                should_include = False
            
            # Add to results if included
            if should_include:
                results[classification]['count'] += 1
                results[classification]['images'].append(image_name)
        
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid JSON in {json_file}: {e}")
            error_count += 1
        except Exception as e:
            print(f"⚠️  Error processing {json_file}: {e}")
            error_count += 1
    
    print(f"{'─'*80}\n")
    
    # Print results
    print(f"{'='*80}")
    print(f"FILTERING RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Total JSON Files: {all_files_count}")
    print(f"Processing Errors: {error_count}")
    print(f"Successfully Processed: {all_files_count - error_count}\n")
    
    # Sort by count (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print(f"Classification Breakdown:\n")
    total_filtered = 0
    for class_name, class_data in sorted_results:
        count = class_data['count']
        total_filtered += count
        print(f"📌 {class_name}")
        print(f"   Count: {count}")
        print(f"   Images:")
        
        # Show first 5 images, then summary
        for i, img in enumerate(class_data['images'][:5], 1):
            print(f"     {i}. {img}")
        
        if len(class_data['images']) > 5:
            remaining = len(class_data['images']) - 5
            print(f"     ... and {remaining} more")
        
        print()
    
    print(f"{'─'*80}\n")
    print(f"Total Filtered Images: {total_filtered}/{all_files_count - error_count}")
    
    # Save to file if requested
    if output_file:
        output_data = {
            'summary': {
                'total_json_files': all_files_count,
                'processing_errors': error_count,
                'successfully_processed': all_files_count - error_count,
                'total_filtered': total_filtered,
                'target_classes': target_classes if target_classes else None,
                'exclude_classes': exclude_classes if exclude_classes else None
            },
            'results': dict(results)
        }
        
        # Convert defaultdict to regular dict for JSON serialization
        output_data['results'] = {k: v for k, v in output_data['results'].items()}
        
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Results saved to: {output_file}")
    
    print(f"\n{'='*80}\n")
    
    return dict(results)


def export_image_paths(results_folder, target_classes, output_file, 
                      exclude_classes=None, with_extensions=True):
    """
    Export filtered image paths as a simple JSON list (for use as input to other scripts).
    
    Args:
        results_folder: Folder containing model output JSON files
        target_classes: List of classes to filter for
        output_file: Path to save the filtered image list
        exclude_classes: List of classes to exclude
        with_extensions: Whether to include file extensions in output
    """
    
    if not os.path.isdir(results_folder):
        raise ValueError(f"Results folder not found: {results_folder}")
    
    exclude_classes = exclude_classes or []
    exclude_classes = [c.lower() for c in exclude_classes]
    target_classes = [c.lower() for c in target_classes]
    
    filtered_images = []
    
    print(f"\n{'='*80}")
    print(f"EXPORT FILTERED IMAGE PATHS")
    print(f"{'='*80}\n")
    
    print(f"📁 Results Folder: {results_folder}")
    print(f"🎯 Target Classes: {target_classes}")
    print(f"💾 Output File: {output_file}")
    print()
    
    # Find and filter JSON files
    json_files = [f for f in os.listdir(results_folder) if f.endswith('.json')]
    
    for json_file in json_files:
        json_path = os.path.join(results_folder, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            classification = data.get('classification', 'unknown').lower()
            
            # Check filters
            if classification in target_classes and classification not in exclude_classes:
                image_name = json_file.replace('_classification.json', '').replace('.json', '')
                filtered_images.append(image_name)
        
        except Exception as e:
            pass  # Skip on error
    
    # Save as JSON list
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_images, f, indent=2)
    
    print(f"✓ Exported {len(filtered_images)} image paths")
    print(f"✓ Saved to: {output_file}\n")
    print(f"{'='*80}\n")
    
    return filtered_images


def get_classification_stats(results_folder):
    """
    Get statistics of all classifications in the results folder.
    
    Args:
        results_folder: Folder containing model output JSON files
    
    Returns:
        Dictionary with classification statistics
    """
    
    stats = defaultdict(int)
    total_files = 0
    error_count = 0
    
    json_files = [f for f in os.listdir(results_folder) if f.endswith('.json')]
    
    for json_file in json_files:
        total_files += 1
        json_path = os.path.join(results_folder, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            classification = data.get('classification', 'unknown')
            stats[classification] += 1
        
        except Exception:
            error_count += 1
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION STATISTICS")
    print(f"{'='*80}\n")
    
    print(f"Total Files: {total_files}")
    print(f"Processing Errors: {error_count}")
    print(f"Unique Classes: {len(stats)}\n")
    
    print(f"Classification Distribution:\n")
    
    # Sort by count
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, count in sorted_stats:
        percentage = (count / total_files * 100) if total_files > 0 else 0
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"{class_name:30} : {count:>6} ({percentage:>5.1f}%) {bar}")
    
    print(f"\n{'='*80}\n")
    
    return dict(stats)


if __name__ == "__main__":
    
    # ========== CONFIGURATION ==========
    
    results_folder = "/datadrive2/IDF_AL_MASRAF/LC_Drawing_Full_Result"
    
    # ========== OPTION 1: Filter and display results ==========
    # Get statistics first
    get_classification_stats(results_folder)
    
    # Filter by specific classes
    filter_images_by_classification(
        results_folder=results_folder,
        target_classes=['Bill_of_Exchange', 'Commercial_Invoice'],
        exclude_classes=['others', 'error'],
        output_file="/datadrive2/IDF_AL_MASRAF/filtered_results.json"
    )
    
    # ========== OPTION 2: Export filtered image paths as list ==========
    # Useful for piping into other processing scripts
    export_image_paths(
        results_folder=results_folder,
        target_classes=['Bill_of_Lading', 'Certificate_of_Origin'],
        output_file="/datadrive2/IDF_AL_MASRAF/filtered_image_paths.json",
        exclude_classes=['others']
    )
    
    # ========== OPTION 3: Filter with NO target (returns all) ==========
    # all_results = filter_images_by_classification(
    #     results_folder=results_folder,
    #     target_classes=None,
    #     exclude_classes=['error'],
    #     output_file="/datadrive2/IDF_AL_MASRAF/all_results.json"
    # )