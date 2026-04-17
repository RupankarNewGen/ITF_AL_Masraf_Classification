import json
import numpy as np
import pytesseract
from PIL import Image
import os
from google.cloud import vision
import json

credentials_path = "/home/ng6309/datascience/hridesh/gvkey.json"

class ApplyOcr:
    def __init__(self):
        # self.apply_ocr()
        self.apply_ocr_gv()


    def apply_ocr_tess(self):
        def normalize_box(box, width, height):
            return [
                int(1000 * (box[0] / width)),
                int(1000 * (box[1] / height)),
                int(1000 * (box[2] / width)),
                int(1000 * (box[3] / height)),
            ]

        # get the image
        c = 0
        image = Image.open(self['image_path'])
        name = self['image_path'].split("/")[-1].split(".")[0]
        width, height = image.size

        # apply ocr to the image
        ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)

        # get the words and actual (unnormalized) bounding boxes
        # words = [word for word in ocr_df.text if str(word) != 'nan'])
        words = list(ocr_df.text)
        words = [str(w) for w in words]
        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        actual_boxes = []
        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
            actual_box = [x, y, x + w,
                          y + h]  # we turn it into (left, top, left+width, top+height) to get the actual box
            actual_boxes.append(actual_box)

        # normalize the bounding boxes
        boxes = []
        for box in actual_boxes:
            boxes.append(normalize_box(box, width, height))

        ocr_data = {}
        # Iterate over the OCR words and boxes
        for word, box in zip(words, boxes):
            box_list = [int(val) for val in box]  # Convert tuple to list
            ocr_data[word] = box_list
        # Convert dictionary to JSON
        json_data = json.dumps(ocr_data)
        print("---------- json_data ---------------")
        return json_data

    def apply_ocr_gv(self,ocr_path):
        def normalize_box(box, width, height):
            return [
                int(1000 * (box[0] / width)),
                int(1000 * (box[1] / height)),
                int(1000 * (box[2] / width)),
                int(1000 * (box[3] / height)),
            ]
        
        # get the image
        c = 0
        image = Image.open(self['image_path'])
        with open(self['image_path'], 'rb') as image_file:
            content = image_file.read()
        name = self['image_path'].split("/")[-1].split(".")[0]

        directory, base_name = os.path.split(self['image_path'])
        base_name_without_ext, _ = os.path.splitext(base_name)
        ocr_json_path = os.path.join(ocr_path,base_name_without_ext+".json")
        ocr_json_text_path = os.path.join(ocr_path,base_name_without_ext+"_text.txt")
        if not os.path.exists(ocr_path):
            os.mkdir(ocr_path)

        width, height = image.size
        image_context = {"language_hints": ["en"]}
        word_coordinates = []
        words_list = []
        corrds_list = []
        if not os.path.exists(ocr_json_path):
            # print("Performing OCR on {}".format(base_name_without_ext))
            client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)
            image = vision.Image(content=content)
            all_text = ""
            response = client.document_text_detection(image=image, image_context=image_context)
            ocr_data = {}
            for i, text in enumerate(response.text_annotations):
                if i != 0:
                    # Extract word coordinates
                    vertices = text.bounding_poly.vertices
                    x1 = min(vertex.x for vertex in vertices)
                    y1 = min(vertex.y for vertex in vertices)
                    x2 = max(vertex.x for vertex in vertices)
                    y2 = max(vertex.y for vertex in vertices)
                    word_coordinates.append({
                        "word": text.description,
                        "left": x1,
                        "top": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
                    ocr_data[text.description] = normalize_box([x1, y1, x2, y2], width, height)
                    words_list.append(text.description)
                    corrds_list.append(normalize_box([x1, y1, x2, y2], width, height))
                else:
                    all_text = text.description
                    

            data={"all_text":all_text,"word_coordinates":word_coordinates,"words_list":words_list,"corrds_list":corrds_list}
            with open(os.path.join(ocr_path,base_name_without_ext+".json"), 'w') as f:
                        json.dump(data, f, indent=4)

            return words_list,corrds_list

        elif os.path.exists(ocr_json_path):
            # print("OCR Exists of {}".format(base_name_without_ext))
            with open(os.path.join(ocr_path,base_name_without_ext+".json"), 'r') as f:
                ocr_info = json.load(f)
            return ocr_info["words_list"],ocr_info["corrds_list"]

        elif os.path.exists(ocr_json_text_path):
            # print("OCR Exists of {}".format(base_name_without_ext))
            with open(os.path.join(ocr_path,base_name_without_ext+".json"), 'r') as f:
                ocr_info = json.load(f)
            ocr_data = {}
            all_word_cords = ocr_info["word_coordinates"]

            for words_info in all_word_cords:
                words_list.append(words_info["word"])
                corrds_list.append(normalize_box([words_info['x1'], words_info['y1'], words_info['x2'], words_info['y2']], width, height))
            return words_list,corrds_list



