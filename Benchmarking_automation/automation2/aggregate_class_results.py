import os
import glob
import pandas as pd
import json

config_path = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(config_path, "r") as f:
        config = json.load(f)
    BENCHMARKING_DIR = config["directories"]["WORKING_DATA_DIR"]
    SPLIT_RESULTS_DIR = config["directories"]["SPLIT_RESULTS_DIR"]
except Exception:
    BENCHMARKING_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/New_test_data_jsons_results"
    SPLIT_RESULTS_DIR = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/New_test_data_jsons_splitted"

OUTPUT_DIR = os.path.join(BENCHMARKING_DIR, "class_wise_aggerate_results")

def aggregate_class_reports(csv_suffix, final_suffix_name):
    # Search for all generated workitem CSVs matching the requested suffix
    search_pattern = os.path.join(BENCHMARKING_DIR, "*", f"*{csv_suffix}")
    all_csvs = glob.glob(search_pattern)
    
    if not all_csvs:
        print(f"No CSVs found matching pattern: {search_pattern}")
        return

    # Load and combine all the CSVs
    all_rows = []
    for csv_file in all_csvs:
        try:
            df = pd.read_csv(csv_file)
            # Drop the local OVERALL_DOCUMENT aggregate row to avoid double-counting
            df = df[df["File_Name"] != "OVERALL_DOCUMENT"]
            all_rows.append(df)
        except Exception as e:
            print(f"Failed to load {csv_file}: {e}")
            
    if not all_rows:
        return
        
    master_df = pd.concat(all_rows, ignore_index=True)
    
    # Split the master dataframe apart entirely by Class_Name natively
    for class_name, class_df in master_df.groupby("Class_Name"):
        if class_name == "UNKNOWN":
            continue
            
        print(f"Processing Class: {class_name} ({len(class_df)} pages found)")
        
        # Calculate the grand total aggregation for this specific class
        total_present = class_df["Total_Fields_Present"].sum()
        total_matched = class_df["Total_Fields_Matched"].sum()
        overall_acc = (total_matched / total_present * 100) if total_present > 0 else 0
        
        # Initialize the export list natively using Python for clean row appending
        export_data = class_df.values.tolist()
        
        # Append the final massive unifying row
        export_data.append(["OVERALL_CLASS_AGGREGATE", class_name, total_present, total_matched, overall_acc])
        
        # Re-wrap and export
        final_df = pd.DataFrame(export_data, columns=["File_Name", "Class_Name", "Total_Fields_Present", "Total_Fields_Matched", "Accuracy_Percentage"])
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        export_path = os.path.join(OUTPUT_DIR, f"{class_name}_{final_suffix_name}")
        final_df.to_csv(export_path, index=False)
        print(f"  -> Generated: {export_path}")

def generate_document_level_summary():
    classification_csv_path = os.path.join(SPLIT_RESULTS_DIR, "final_aggregation_report.csv")
    
    if not os.path.exists(classification_csv_path):
        print(f"Classification report not found at {classification_csv_path}")
        return
        
    class_df = pd.read_csv(classification_csv_path)
    
    # Ensure mapping of class results to lookup by folder name
    class_df['folder name'] = class_df['folder name'].astype(str)
    
    summary_data = []
    
    # Iterate through all document folders in BENCHMARKING_DIR (like DOC_3)
    document_folders = [d for d in os.listdir(BENCHMARKING_DIR) if os.path.isdir(os.path.join(BENCHMARKING_DIR, d)) and d != "class_wise_aggerate_results"]
    
    for doc_name in document_folders:
        doc_folder_path = os.path.join(BENCHMARKING_DIR, doc_name)
        extraction_csv_path = os.path.join(doc_folder_path, f"{doc_name}_final_document_page_wise_summary.csv")
        
        if not os.path.exists(extraction_csv_path):
            continue
            
        ext_df = pd.read_csv(extraction_csv_path)
        # Find the overall document aggregate row
        overall_row = ext_df[ext_df["File_Name"] == "OVERALL_DOCUMENT"]
        if overall_row.empty:
            continue
            
        total_fields = overall_row["Total_Fields_Present"].values[0]
        correct_fields = overall_row["Total_Fields_Matched"].values[0]
        extraction_acc = overall_row["Accuracy_Percentage"].values[0]
        
        # Format the extraction accuracy string if it's purely a float
        try:
            extraction_acc = f"{float(extraction_acc):.2f}%"
        except:
            pass
        
        # Look up classification details
        class_row = class_df[class_df["folder name"] == doc_name]
        
        if class_row.empty:
            total_pages = 0
            correct_classification = 0
            classification_accuracy = "0.00%"
        else:
            total_pages = class_row["total_page"].values[0]
            correct_classification = class_row["correct_page"].values[0]
            classification_accuracy = class_row["accuracy"].values[0]
            
        summary_data.append([
            doc_name, 
            total_fields, 
            correct_fields, 
            extraction_acc, 
            total_pages, 
            correct_classification, 
            classification_accuracy
        ])
        
    if not summary_data:
        print("No document summaries were generated. Please check files and paths.")
        return
        
    final_df = pd.DataFrame(summary_data, columns=[
        "document_name", "total fields", "correct_fields", "accuracy",
        "total_pages", "correct_classification", "classification_accuracy"
    ])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_export_path = os.path.join(BENCHMARKING_DIR, "final_document_level_summary.csv")
    final_df.to_csv(summary_export_path, index=False)
    print(f"\n✅ Document-level summary created at: {summary_export_path}")

def main():
    print("Aggregate 1: Original Config Thresholds")
    aggregate_class_reports("_final_document_page_wise_summary.csv", "overall_pagewise_summary_native.csv")
    
    print("\nAggregate 2: Uniform Fuzzy Thresholds")
    # Will match whatever _final_document_page_wise_summary_*.csv is there
    # Assuming standard pattern, this searches for any numbers:
    aggregate_class_reports("_final_document_page_wise_summary_*.csv", "overall_pagewise_summary_threshold.csv")
    
    print("\nAggregate 3: Master Document-Level Summary")
    generate_document_level_summary()

if __name__ == "__main__":
    main()
