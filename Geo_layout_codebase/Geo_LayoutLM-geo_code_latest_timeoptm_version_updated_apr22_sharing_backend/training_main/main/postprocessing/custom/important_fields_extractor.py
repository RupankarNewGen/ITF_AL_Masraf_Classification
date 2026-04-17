import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    #varaiable definition
    root= '/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/CS_complete_data/new_data_prep/ROOT_CS'
    report_path = f"{root}/result_path/final_report_PL_pre_May11_after_fuzzy_match_change.csv" #  csv file generated after accuracy gen script
    analysis_report_path = f"{root}/result_path/CS_analysis_pre_valid_sep_12_after_fuzzy_match_post_processing_latest.csv"
    folder_path= root
    accuracy_lookup = pd.read_csv(report_path)
    analysis_report = pd.read_csv(analysis_report_path)
    data = []
    label_names= []

    #final_Report_Summary_Best_Fields
    for index, row in accuracy_lookup.iterrows():
        if row["Complete_Match_Percentage"] >= 80 and row["Fuzzy_Match_Percentage"] >= 85 and row["Label_Name"] != "OVERALL": #change these metrics to create buckets.
            label_names.append(str(row["Label_Name"]))
            data.append(list(row)[1:])
    res_path= os.path.join(folder_path, 'result_path')
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    save_path_1 = "Final_Report_Summary_Best_Fields_after_fuuzy_match_change.csv"
    save_path_1= os.path.join(res_path, save_path_1)
    df2 =  pd.DataFrame(data, columns = list(accuracy_lookup.columns)[1:])
    df2.to_csv(save_path_1)

    new_data = []
    #train-test_important_labels
    for index, row in analysis_report.iterrows():
        if str(row["label_name"]) in label_names:
            new_data.append(list(row)[1:])
    name= "Overall_Analysis_Best_Fields_after_fuzzy_match_change.csv"
    save_path_2= os.path.join(res_path, name)
    df2 =  pd.DataFrame(new_data, columns = list(analysis_report.columns)[1:])
    df2.to_csv(save_path_2)

