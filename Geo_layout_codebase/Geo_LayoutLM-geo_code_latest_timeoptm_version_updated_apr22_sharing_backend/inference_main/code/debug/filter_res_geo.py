import pandas as pd
import numpy as np

def process_labels_and_scores(label_file_path, csv_files_dict, output_file_path):
    """
    Process multiple CSV files provided in a dictionary and combine their label matching percentages
    
    Args:
        label_file_path (str): Path to the label.txt file
        csv_files_dict (dict): Dictionary with keys as identifiers and values as file paths
        output_file_path (str): Path where the result CSV will be saved
    """
    # Read labels from txt file
    with open(label_file_path, 'r') as file:
        labels = [line.strip() for line in file if line.strip()]
    labels = list(set(labels))
    print(labels)
    print(len(labels))
    # Create a dictionary to store results for each file
    all_results = {}
    
    # Process each CSV file
    for file_key, csv_file in csv_files_dict.items():
        # try:
        df = pd.read_csv(csv_file)
        
        # Create a dictionary for this file's matching percentages
        matching_dict = {}
        
        # For each label from txt file, find its matching percentage or assign NaN
        for label in labels:
            print('START >>', label)
            matching_row = df[df['Label_Name'] == label]
            
            if not matching_row.empty:
                matching_dict[label] = matching_row['Complete_Match_Percentage'].iloc[0]
            else:
                matching_dict[label] = np.nan
            print('END >>', matching_dict[label])
        all_results[file_key] = matching_dict
            
        # except Exception as e:
        #     print(f"Error processing file for key {file_key}: {str(e)}")
        #     continue
    print(all_results)
    # Create a DataFrame with all results
    result_df = pd.DataFrame.from_dict(all_results, orient='columns')
    print(len(labels))
    print(len(result_df))

    # Add Label_Name as the first column
    result_df.insert(0, 'Label_Name', labels)
    
    # Save to CSV
    result_df.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")
    

    
    return result_df

# Example usage
label_file = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/BOL_EVAL/v1/label.txt"
#CI
# csv_files = {
#     "geo_ci_v1": "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/CI/v1/result_path_with_aug/label_wise/ci_2024-12-30_18/final_report_ci_pre_2024-12-30 18:16:14.509540_after_fuzzy_match_change.csv",
#     "geo_ci__v1_without_aug": "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/CI/v1/result_path_without_aug/label_wise/ci_2025-01-14_16/final_report_ci_pre_2025-01-14 16:32:17.888039_after_fuzzy_match_change.csv",
#     "geo_ci_v2": "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/CI/v2/result_path_with_aug/label_wise/ci_2025-01-13_16/final_report_ci_pre_2025-01-13 16:03:47.038019_after_fuzzy_match_change.csv",
#     "geo_ci_v2_without_aug": "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/CI/v2/result_path_without_aug/label_wise/ci_2025-01-15_12/final_report_ci_pre_2025-01-15 12:48:36.367709_after_fuzzy_match_change.csv",
#     "lmv2": "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/CI/v1/final_report_lmv2.csv"
# }
#BOL
csv_files = {
    "lmv2": "/home/ntlpt19/Downloads/Evaluation_Data/FinalEvaluationEvalData/BOL/result_path_final_cate_wise/label_wise/bol_2024-02-22_11/final_report_bol_pre_2024-02-22 11:26:05.627722_after_fuzzy_match_change.csv",
    "geo_bol_v2_without_aug": "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/BOL_EVAL/v1/result_path/label_wise/bol_2025-01-17_12/final_report_bol_pre_2025-01-17 12:07:05.511735_after_fuzzy_match_change.csv",
}

output_file = "combined_matching_percentages.csv"

results = process_labels_and_scores(label_file, csv_files, output_file)
    
    
