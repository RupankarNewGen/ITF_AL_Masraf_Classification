import json
import os
from collections import defaultdict

def audit_unique_leaf_counts(json_paths):
    """
    Groups data by category, counts unique leaf IDs per category, 
    and provides image counts per leaf ID.
    """
    # Categorization structure
    categories = ["Import LC Acceptance", "Import LC Drawing", "Import LC Settlement"]
    
    # Store unique leaf IDs per category: { "Category": set(ID1, ID2...) }
    category_leaf_map = {cat: set() for cat in categories}
    
    # Store image counts per leaf: { "LeafID": count }
    leaf_image_counts = defaultdict(int)

    # 1. Parse Manifests
    for path in json_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for img_path in data:
            # Extract leaf folder name
            parent_dir = os.path.dirname(img_path)
            leaf_id = os.path.basename(parent_dir)
            
            # Map leaf ID to category
            for cat in categories:
                if cat in img_path:
                    category_leaf_map[cat].add(leaf_id)
            
            # Count images per leaf
            leaf_image_counts[leaf_id] += 1

    # 2. Print Summary Report
    print("="*60)
    print(f"{'CATEGORY':<30} | {'UNIQUE LEAF COUNT':<20}")
    print("-" * 60)
    
    for cat in categories:
        unique_count = len(category_leaf_map[cat])
        print(f"{cat:<30} | {unique_count:<20}")
        # Uncomment the line below if you want to see the specific IDs under each category
        # print(f"   IDs: {sorted(list(category_leaf_map[cat]))}\n")

    print("\n" + "="*60)
    print(f"{'LEAF ID':<30} | {'TOTAL IMAGES':<20}")
    print("-" * 60)
    
    # Sort by ID for easier reading
    for leaf in sorted(leaf_image_counts.keys()):
        print(f"{leaf:<30} | {leaf_image_counts[leaf]:<20}")

    print("="*60)

if __name__ == "__main__":
    # Add your list of manifest JSON files here
    INPUT_JSONS = [
        "/datadrive2/IDF_AL_MASRAF/targeted_sibling_expansion.json",
        "/datadrive2/IDF_AL_MASRAF/remaining_untapped_images.json",
        "/datadrive2/IDF_AL_MASRAF/final_balanced_sample.json"
    ]

    audit_unique_leaf_counts(INPUT_JSONS)