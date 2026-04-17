import os
import glob
import pandas as pd

WORKSPACE_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/missclassified_images_benchmarking/DOC_1"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "Master_Class_Aggregations")

def process_page_wise(csv_pattern, output_suffix):
    search_pattern = os.path.join(WORKSPACE_DIR, "*", "result_path", "*", csv_pattern)
    native_csvs = glob.glob(search_pattern)
    
    if not native_csvs:
        print(f"No files found for pattern: {search_pattern}")
        return

    # Group CSVs by their parent class folder
    class_map = {}
    for csv_file in native_csvs:
        # Path structure: WORKSPACE_DIR/ClassName/result_path/timestamp/file.csv
        parts = csv_file.split(os.sep)
        try:
            result_path_idx = parts.index("result_path")
            class_name = parts[result_path_idx - 1]
        except ValueError:
            class_name = "UNKNOWN"
            
        if class_name not in class_map:
            class_map[class_name] = []
        class_map[class_name].append(csv_file)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    for class_name, csv_files in class_map.items():
        print(f"Processing {class_name} for {output_suffix} ({len(csv_files)} files)")
        
        df_list = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df_list.append(df)
            except Exception as e:
                print(f"  -> Failed to read {csv_file}: {e}")
                
        if not df_list:
            continue
            
        master_df = pd.concat(df_list, ignore_index=True)
        master_df['Class_Name'] = class_name
        
        page_data = []
        for file_name, file_df in master_df.groupby("File_Name"):
            # Drop empty/NaN ground truth fields
            valid_df = file_df.dropna(subset=['actual']).copy()
            valid_df = valid_df[valid_df['actual'].astype(str).str.strip() != ""]
            valid_df = valid_df[valid_df['actual'].astype(str).str.strip().str.lower() != "nan"]
            
            total_fields = len(valid_df)
            valid_df["Match/No_Match"] = pd.to_numeric(valid_df["Match/No_Match"], errors='coerce').fillna(0)
            matched_fields = valid_df["Match/No_Match"].sum()
            
            page_acc = (matched_fields / total_fields * 100) if total_fields > 0 else 0
            page_data.append([file_name, class_name, total_fields, matched_fields, page_acc])
            
        # Calculate single overall class aggregate
        class_total_present = sum(item[2] for item in page_data)
        class_total_matched = sum(item[3] for item in page_data)
        class_acc = (class_total_matched / class_total_present * 100) if class_total_present > 0 else 0
        
        page_data.append(["OVERALL_CLASS_AGGREGATE", class_name, class_total_present, class_total_matched, class_acc])
        
        summary_df = pd.DataFrame(page_data, columns=["File_Name", "Class_Name", "Total_Fields_Present", "Total_Fields_Matched", "Accuracy_Percentage"])
        export_path = os.path.join(OUTPUT_DIR, f"{class_name}_{output_suffix}")
        summary_df.to_csv(export_path, index=False)
        print(f"  -> Generated: {export_path}")

def main():
    print("Aggregate 1: Original Config Thresholds")
    process_page_wise("*_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv", "misclassified_page_wise_native.csv")
    
    print("\nAggregate 2: Uniform 75% Thresholds")
    process_page_wise("*_analysis_fuzzy_75.csv", "misclassified_page_wise_75.csv")

if __name__ == "__main__":
    main()
