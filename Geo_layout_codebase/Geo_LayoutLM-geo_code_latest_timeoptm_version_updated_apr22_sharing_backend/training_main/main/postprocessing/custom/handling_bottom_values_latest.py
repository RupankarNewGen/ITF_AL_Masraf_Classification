import os
import json
from datetime import datetime
start_time = datetime.now()

	# Get the file name of the executed Python script
#file_name = os.path.basename(__file__)
def check_and_modify_labels(json_data, label):
    if label in json_data and bottom_value in json_data:
        json_data[bottom_value][0][0] = "~ "+json_data[bottom_value][0][0]
        
        #joined_value = "~ ".join([json_data[label][0][0], json_data["insurance_issuer_address_bottom"][0][0]])
        joined_value = json_data[label]+json_data[bottom_value]
        json_data[label] = joined_value
        del json_data[bottom_value]
    else:
        if bottom_value in json_data:
            json_data[label] = json_data.pop(bottom_value)

    return json_data

folder_path: str = r"/home/ntlpt19/Downloads/Trade_finance_imp_stage_2/IC_ROOT/IC_ROOT_MERGE"
top_value = "insurance_issuer_name"
bottom_value = "insurance_issuer_name_bottom"



result_path = os.path.join(folder_path, "Results_CS_validated")
data_path = os.path.join(folder_path, "New_Master_Data_Merged")

# import json
data_files = os.listdir(data_path)
result_files = os.listdir(result_path)
for file in data_files:
        print("file name is:", file)
        # print("resulted filename is", predicted_files)
        # continue
        with open(os.path.join(data_path, file), "r") as f:
            file_path = os.path.join(data_path, file)
            labels = json.load(f)
            modified_data = check_and_modify_labels(labels, top_value)
            with open(file_path, "w") as modified_file:
                json.dump(modified_data, modified_file)

for file in result_files:
        print("file name is:", file)
        # print("resulted filename is", predicted_files)
        # continue
        try:
            print(os.path.join(result_path, file[0:-5] + "1.txt"))
            with open(os.path.join(result_path, file[0:-5] + "1.txt"), "r") as f2:
                file_result_path = os.path.join(result_path, file[0:-5] + "1.txt")
                predicted = json.load(f2)
                modified_data = check_and_modify_labels(predicted, top_value)
                with open(file_result_path, "w") as modified_file:
                    json.dump(modified_data, modified_file)
        except:
        # print("some problem opening file")
            try:
                print(f'''printing the path: {result_path, file[0:-5] + "_s_11.txt"}''')
                with open(os.path.join(result_path, file[0:-5] + "_s_11.txt"), "r") as f2:
                    file__result_path = os.path.join(result_path, file[0:-5] + "_s_11.txt")
                    predicted = json.load(f2)
                    modified_data = check_and_modify_labels(predicted, top_value)
                    with open(file__result_path, "w") as modified_file:
                        json.dump(modified_data, modified_file)
            # print("opened")
            except:
                print("still not opened")
                continue
