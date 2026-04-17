
import pandas as pd

import os

if __name__== "__main__":
    folder_path= "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/Layoutlmv3_performance_phase_1_data/CS_jul_13/evaluation_data"
    file_name= 'result_path/final_report_PL_pre_May11_after_fuzzy_match_change.csv'
    best_fields_path= 'best_fields'
    best_fields=[]
    with open(f'{folder_path}/{best_fields_path}') as f:
        for label in f.readlines():
            best_fields.append(label.strip())
    print(best_fields)
    # exit('+++++++')
    result_path= 'best_fields_based_on_evaluation.csv'
    # images_path= os.path.join(folder_path, 'Images')
    # images_lst= os.listdir(images_path)
    df= pd.read_csv(f'{folder_path}/{file_name}')
    print(df.columns)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    # exit('+++++++')
    # print(len(images_lst)//100*40)
    filtered_df = df[df['Label_Name'].isin(best_fields)]
    filtered_df= filtered_df.reset_index()
    print(filtered_df)
    filtered_df.to_csv(f'{folder_path}/{result_path}', index=True)
   

    # labels_lst= df['Label_Name'].to_list()
    # print(labels_lst)
    # # exit('+++++++++++')
    # print('dataframe saved successfully!')
    # # print(filtered_df)
    # eval_df_path= '/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/stp_verification_on_100_samples/PackingList/iteration_2/result_path/final_report_PL_pre_june_19_after_fuzzy_match_change.csv'
    # df= pd.read_csv(eval_df_path)
    # df.drop('Unnamed: 0', axis=1, inplace=True)
    # filtered_df = df[df['Label_Name'].str.contains('|'.join(labels_lst))]
    # filtered_df.to_csv('/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/stp_verification_on_100_samples/PackingList/iteration_2/result_path/best_fields_of_evaluation_data.csv', index=True)


