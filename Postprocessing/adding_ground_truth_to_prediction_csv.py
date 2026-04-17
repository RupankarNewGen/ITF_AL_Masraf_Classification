import pandas as pd
import sys

def sync_ground_truth(first_csv_path, second_csv_path, output_csv_path):
    """
    Reads two CSVs, matches them on 'image name', and copies the 'Ground Truth'
    column from the first to the second. Handles Windows encoding errors.
    """
    
    def robust_read_csv(path):
        """Helper to try common encodings for Windows-generated CSVs."""
        for enc in ['utf-8', 'cp1252', 'latin1', 'utf-16']:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode the file at {path} with common encodings.")

    try:
        # 1. Load both CSV files using the robust loader
        df1 = robust_read_csv(first_csv_path)
        df2 = robust_read_csv(second_csv_path)

        # 2. Clean 'image name' columns (remove leading/trailing spaces)
        # This prevents "image1.jpg" failing to match " image1.jpg"
        if 'image name' in df1.columns:
            df1['image name'] = df1['image name'].astype(str).str.strip()
        if 'image name' in df2.columns:
            df2['image name'] = df2['image name'].astype(str).str.strip()

        print(f"Loaded CSV 1: {len(df1)} rows")
        print(f"Loaded CSV 2: {len(df2)} rows")

        # 3. Validation: Check if columns exist
        if 'image name' not in df1.columns or 'Ground Truth' not in df1.columns:
            print("Error: CSV 1 must have 'image name' and 'Ground Truth' columns.")
            return
        if 'image name' not in df2.columns:
            print("Error: CSV 2 must have 'image name' column.")
            return

        # 4. Prepare mapping data (remove duplicates to avoid row explosion)
        mapping_subset = df1[['image name', 'Ground Truth']].drop_duplicates(subset=['image name'])

        # 5. Merge the data (Left Join)
        # This keeps all rows in df2 and adds 'Ground Truth' where a match exists
        updated_df = pd.merge(df2, mapping_subset, on='image name', how='left')

        # 6. Save the result
        # utf-8-sig ensures Excel handles special characters correctly
        updated_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

        # 7. Final Statistics
        matched_count = updated_df['Ground Truth'].notna().sum()
        unmatched_count = updated_df['Ground Truth'].isna().sum()
        
        print("\n" + "="*40)
        print("SYNC COMPLETE")
        print("="*40)
        print(f"Total rows processed:  {len(df2)}")
        print(f"Matches found:         {matched_count}")
        print(f"No match found:        {unmatched_count}")
        print(f"File saved to:         {output_csv_path}")
        print("="*40)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # First CSV: The source of truth
    FILE_SOURCE = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Prediction_CSV/second_set_1000_images_results(in).csv"
    
    # Second CSV: The file you want to update
    FILE_TARGET = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Prediction_CSV/second_set_1000_images_results.csv"
    
    # Output path
    FILE_OUTPUT = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/Prediction_CSV/second_set_1000_images_with_ground_truth.csv"

    sync_ground_truth(FILE_SOURCE, FILE_TARGET, FILE_OUTPUT)