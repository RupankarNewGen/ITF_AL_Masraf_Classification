import os
import json


def remove_spaces_in_keys(labels):
    keys_with_spaces = list(labels.keys())
    for key in keys_with_spaces:
        if " " in key:
            new_key = key.replace(" ", "")
            labels[new_key] = labels.pop(key)
    return labels





if __name__=="__main__":
    folder_path: str = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/Layoutlmv3_performance_phase_1_data/CS_july_31_phase_2/validation_data"


    master_data_path= os.path.join(folder_path, "New_Master_Data_Merged")


    image_files = os.listdir(master_data_path)
    image_files = [x.split("_labels.txt")[0] for x in image_files]

    print(image_files)
	# exit('+++++++++++=')

    for file in image_files:

        with open(os.path.join(master_data_path, file+"_labels.txt"), "r") as f:
            labels = json.load(f)
            print(file)
            print(labels)
            # exit('+++++++++++++')
            remove_spaces_in_keys(labels)
            print(labels)
            # exit('+++++++++=')
            print('yes')
            # print(f'''{file.split(".png")[0]}_labels.txt''')
            # exit('++++++++=')
            # print(labels)
        with open(os.path.join(master_data_path, file+'_labels.txt'), "w") as f:
            json.dump(labels, f)
        # exit('+++++++')