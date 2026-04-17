import pandas as pd
import os

if __name__ == "__main__":
	file_name = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/Layoutlmv3_performance_phase_1_data/CS_july_5_phase2_data/Root/result_path/BOL_analysis_pre_valid_May30_after_fuzzy_match_post-processing.csv"
	folder_path= "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/Layoutlmv3_performance_phase_1_data/CS_july_5_phase2_data/Root"
	df = pd.read_csv(file_name)
	# test_df = df[(df["Train/Test"] == "Test") |  (df["Train/Test"] == "Both")]
	g = df.groupby("File_Name")
	stp_data = []
	num_files = 0
	match_files = 0
	for name, name_df in g:
		num_files += 1
		match_value = sum(list(name_df["Match/No_Match"]))
		num_labels = len(list(name_df["Match/No_Match"]))
		if match_value == num_labels:
			match_files += 1
			flag = 1
		else:
			flag = 0
		row = [str(name), match_value, num_labels, flag]
		stp_data.append(row)
	print("stp is", (match_files / num_files) * 100)

	df2 = pd.DataFrame(stp_data, columns=["File_Name", "Labels_Matched", "Labels_Present", "STP_Match"])
	res_path= os.path.join(folder_path, 'result_path')
	if not os.path.exists(res_path):
		os.mkdir(res_path)
	name= "STP_Summary_25.csv"
	res_path= os.path.join(res_path, name)
	df2.to_csv(res_path)