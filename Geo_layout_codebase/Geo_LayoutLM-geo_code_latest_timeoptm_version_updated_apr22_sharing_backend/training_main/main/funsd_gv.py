import os
import json
from ast import literal_eval

def gen_custom_data(root_path, ocr_file):
    custom_path = os.path.join(root_path, "custom_data")
    if not os.path.exists(custom_path):
        os.mkdir(custom_path)
    all_words_path = os.path.join(custom_path, "all_words")
    if not os.path.exists(all_words_path):
        os.mkdir(all_words_path)


    # Assuming 'filenames' contains the list of filenames
    filtered_filenames = [
        filename for filename in os.listdir(ocr_file)
        if ('textAndCoordinates' in filename or '_text' in filename) and 'all_text' not in filename
    ]

    for j in filtered_filenames:
        if 'textAndCoordinates' in j:
            file_save_numb = 23
        elif '_text' in j:
            file_save_numb = 9
            
        print(j[0:-file_save_numb])
        # Open the file in read mode ('r')
        if j.endswith('.txt'):
            with open(os.path.join(ocr_file, j), 'r') as file:
                # Read the contents of the file
                # word_coordinates = eval(file.read())
                word_coordinates = literal_eval(file.read())
            print(word_coordinates)
            file.close()
            if isinstance(word_coordinates, dict):
                word_coordinates = word_coordinates.get('word_coordinates', word_coordinates.get('ocrContent', []))

        else:
            with open(os.path.join(ocr_file, j), "r") as f:
                word_coordinates = json.load(f)#['word_coordinates']
            f.close()
        cou = 1
        final = {}
        import json
        for i in word_coordinates:
            final[cou]={'text':i['word'], 'bbox':[i['x1'],i['y1'],i['x2'], i['y2']]}
            cou+=1
        # print(final)
        file_name = os.path.join(all_words_path, j[0:-file_save_numb]+'.json')
        with open(file_name, "w") as json_file:
            json.dump(final, json_file)
            
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Data Generation Script")
    parser.add_argument(
        "--root_path",
        default = '/',
        help="Root directory path",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    
    ROOT_PATH = args.root_path
    # ROOT_PATH = '/root/rakesh/data/BOE'
    OCR_PATH = os.path.join(ROOT_PATH, 'OCR')
    gen_custom_data(ROOT_PATH, OCR_PATH)