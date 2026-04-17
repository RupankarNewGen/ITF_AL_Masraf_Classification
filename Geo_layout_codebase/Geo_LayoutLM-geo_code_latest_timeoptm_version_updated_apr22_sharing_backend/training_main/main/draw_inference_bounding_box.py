import cv2
import os
import json
from typing import Dict

def draw_bounding_box(img_path, json_data:Dict):
    image = cv2.imread(img_path)
    # thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    for item in json_data:
        key_bbox=item['words'][0]['box']
        cv2.rectangle(image, (key_bbox[0], key_bbox[1]), (key_bbox[2], key_bbox[3]), (0, 255, 0), 2)
        cv2.putText(
        image,
        "word",
        (int(key_bbox[0]), int(key_bbox[1])),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.6,
        color = (255, 0, 0),
        thickness=2
    )
    return image


if __name__== '__main__':
    root_path= "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/test_code/Validation_data/test/inference_files"

    image_path = os.path.join(root_path, 'images')
    annot_path = os.path.join(root_path,'annotations')
    bounding_box_path= os.path.join(root_path, "bounding_box")
    if not os.path.exists(bounding_box_path):
        os.mkdir(bounding_box_path)

    img_files= os.listdir(image_path)
    img_files=  [file.split('.png')[0] for file in img_files]
    print(img_files)
    # exit('++++++++++++')
    for file in img_files:
        print(f'processing file: {file}')
        try:
            if os.path.exists(os.path.join(annot_path, file+'.json')):
                with open(os.path.join(annot_path, file+'.json'), 'r') as f:
                    data = json.load(f)['form']
                print(data)
                # exit('++++++++++++++')
                processed_img=draw_bounding_box(os.path.join(image_path, file+'.png'), data)
                bounding_box= os.path.join(bounding_box_path, file+ ".png")
                cv2.imwrite(bounding_box, processed_img)
                # for entry in data:
                #     key_text = " ".join(entry["key_text"])
                #     value_bboxes = entry["value_bbox"]
                #     print(f"key_text: {key_text}, value_bboxes:{value_bboxes}")
                # #     extracted_data.append({"key_text": key_text, "value_bboxes": value_bboxes})

                # print(extracted_data)
        except Exception as e:
            print('Annotation file has missed!')
            continue
        


    



