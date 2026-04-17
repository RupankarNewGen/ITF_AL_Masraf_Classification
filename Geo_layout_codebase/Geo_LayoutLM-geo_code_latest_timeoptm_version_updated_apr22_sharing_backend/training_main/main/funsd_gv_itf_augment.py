import os
import json
from ast import literal_eval

PREFIXES = [
    "Shifted_",
    "hollow_op_",
    "watermark__",
    "dirtydrum_op_",
    "dotmatrix_op_",
    "colorpaper_op_",
    "colorshift_op_",
    "salt_n_pepper_",
    "bleedthrough_op_",
    "dirty_rollers_op_",
    "BadPhotoCopy_new_op_",
    "binder_punch_holes_op_",
    "depthsimulatedblur_op_",
    "brightness_texturize_op_",
    ]
def get_original_image_name(image_name, prefixes):
    for prefix in prefixes:
        if image_name.startswith(prefix):
            # Remove the prefix and return the original name
            return image_name[len(prefix):]
    # Return the original name if no prefix matches
    return image_name


def gen_custom_data(root_path, ocr_file, Images_path):
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
    
    for imgs_ in os.listdir(Images_path):
        print(imgs_)
        image_name = get_original_image_name(imgs_, PREFIXES)
        print(image_name)
        # for j in filtered_filenames    
        j = image_name.replace('.png', '_text.txt')
        print(os.path.join(ocr_file, j))
        # Open the file in read mode ('r')
        if j.endswith('.txt') and os.path.exists(os.path.join(ocr_file, j)):
            with open(os.path.join(ocr_file, j), 'r') as file:
                # Read the contents of the file
                # word_coordinates = eval(file.read())
                word_coordinates = literal_eval(file.read())
            print(word_coordinates)
            file.close()
            if isinstance(word_coordinates, dict):
                word_coordinates = word_coordinates.get('word_coordinates', [])

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
        file_name = os.path.join(all_words_path, imgs_.replace('.png', '.json'))
        with open(file_name, "w") as json_file:
            json.dump(final, json_file)
                

if __name__ == "__main__":
    OCR_PATH = '/datadrive/rakesh/TradeFinanceData/CI/OCR'
    ROOT_PATH = '/datadrive/rakesh/TradeFinanceData/CI'
    Images_path = '/datadrive/rakesh/TradeFinanceData/CI/Images'
    gen_custom_data(ROOT_PATH, OCR_PATH, Images_path)