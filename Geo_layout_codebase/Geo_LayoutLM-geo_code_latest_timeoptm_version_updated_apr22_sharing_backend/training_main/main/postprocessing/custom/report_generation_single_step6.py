import pandas as pd
import json
import os

if __name__ == '__main__':

	folder_path: str = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/CS_complete_data/new_data_prep/ROOT_CS"
	df = pd.read_csv(os.path.join("/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/CS_complete_data/new_data_prep/ROOT_CS/result_path/CS_analysis_pre_valid_sep_12_after_fuzzy_match_post_processing_latest.csv"))

	report = {}
	label_names = []
	label_counts = []
	label_detected = []
	detection_accuracy = []
	average_accuracy = []
	matched_labels = []
	matched_detected = []
	total_match_percentage = []
	g = df.groupby("label_name")
	for name, name_df in g:
		label_names.append(name)
		label_counts.append(len(name_df.index))
		average_accuracy.append(round(name_df["Accuracy"].mean(), 2))
		matched = sum(list(name_df["Match/No_Match"]))
		print(f'matched numbers: {matched}')
		matched_labels.append(matched)
		total_match_percentage.append(round((matched / len(name_df.index)) * 100, 2))
		not_detected = name_df["predicted"].isnull().sum()
		detected = len(name_df.index) - not_detected
		print(f'detected numbers: {detected}')
		label_detected.append(detected)
		matched_detected.append(matched / detected)
		print(matched_detected)
		detection_accuracy.append((detected / len(name_df.index)) * 100)
	data = {'Label_Name': label_names, 'Label_Count': label_counts, "Labels_Detected": label_detected,
	        "Detection_Accuracy": detection_accuracy, "Fuzzy_Match_Percentage": average_accuracy,
	        "Complete_Match_Count_Detected": matched_detected, "Complete_Match_Count": matched_labels,
	        "Complete_Match_Percentage": total_match_percentage}

	# Just to calculate detection accuracy
	detected = sum(label_detected)
	all_matched = sum(matched_labels)
	avg_matched_detected = all_matched / detected
	all_labels = sum(label_counts)
	overall_detection_accuracy = (detected / all_labels) * 100
	# dataframe and csv file generation
	report = pd.DataFrame(data)
	li = ["OVERALL", sum(list(report["Label_Count"])), sum(list(report["Labels_Detected"])), overall_detection_accuracy,
	      df["Accuracy"].mean(), avg_matched_detected, sum(list(report["Complete_Match_Count"])),
	      sum(list(report["Complete_Match_Count"])) / sum(list(report["Label_Count"]))]
	report.loc[len(report.index)] = li
	res_path= os.path.join(folder_path, 'result_path')
	if not os.path.exists(res_path):
		os.mkdir(res_path)
	name = "final_report_PL_pre_May11_after_fuzzy_match_change" + ".csv"
	name_path= os.path.join(res_path, name)
	report.to_csv(name_path)

	# txt file generation

	name = "final_report_" + ".txt"
	name_path= os.path.join(res_path, name)
	text_report = {}
	for index, row in report.iterrows():
		text_report[row["Label_Name"]] = {"Label_Count": row["Label_Count"],
		                                  "Detection_Accuracy": row["Detection_Accuracy"],
		                                  "Fuzzy_Match_Percentage": row["Fuzzy_Match_Percentage"],
		                                  "Complete_Match_Count_Detected": row["Complete_Match_Count_Detected"],
		                                  "Complete_Match_Count": row["Complete_Match_Count"],
		                                  "Complete_Match_Percentage": row["Complete_Match_Percentage"]}
	with open(name_path, "w") as f:
		json.dump(text_report, f)