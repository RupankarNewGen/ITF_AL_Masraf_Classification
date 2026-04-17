import os
import cv2
import matplotlib.pyplot as plt
import shutil
import json
from functools import cmp_to_key
from google.cloud import vision
from base64 import b64encode
import time
import pytesseract

zoom = 300/72


def contour_sort(a, b):
    if abs(a['y1'] - b['y1']) <= 15:
        return a['x1'] - b['x1']

    return a['y1'] - b['y1']


def remove_garbage(dataset):
    to_remove = ["\u00da", "\u00c6", "\u00c4", "\u00b4", "\u00c5", "Á","\n","|"]
    for key in dataset.keys():
        values = dataset[key]
        for value in values:
            string = value[0]
            new_string = ""
            for char in string:
                if char not in to_remove:
                    new_string+= char
            new_string = new_string.strip()
            value[0] = new_string        


# def get_ocr_tesseract(img):
# 	print("called OCR tesseract...", end = "")
# 	#img = cv2.imread(image)
# 	#hImg,wImg,_ = img.shape
# 	d = pytesseract.image_to_data(img)
# 	all_text = pytesseract.image_to_string(img)
# 	word_coordinates = []
# 	for i,b in enumerate(d.splitlines()):
# 		if i!=0:
# 			b = b.split()
# 			if len(b) == 12:
# 				word = b[11]
# 				x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
# 				dic = {
# 					"word": word,
# 					"left": x,
# 					"top": y,
# 					"width": w,
# 					"height": h,
# 					"x1": x,
# 					"y1": y,
# 					"x2": x+w,
# 					"y2": y+h,
# 					"coordinates": [x,y,x+w,y+h],
# 					"confidence": float(b[10])
# 					}
# 				if dic not in word_coordinates:
# 					word_coordinates.append(dic)
# 				else:
# 					continue
# 	return word_coordinates, all_text


def get_ocr_vision_api(image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/ntlpt-42/Downloads/oval-heuristic-387906-1815ed6ee296.json"
    with open(image_path, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content = ctxt)

    response = client.text_detection(image=image)

    word_coordinates = []
    all_text = ""

    for i,text in enumerate(response.text_annotations):
        if i != 0:
            # print('=' * 30)
            # print(text.description)
            vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
            x1 = min([v.x for v in text.bounding_poly.vertices])
            x2 = max([v.x for v in text.bounding_poly.vertices])
            y1 = min([v.y for v in text.bounding_poly.vertices])
            y2 = max([v.y for v in text.bounding_poly.vertices])
            # print('bounds: ' + str(vertices))
            if x2 - x1 == 0:
                x2 += 1
            if y2 - y1 == 0:
                y2 += 1
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
        else:
            all_text = text.description

    return word_coordinates, all_text







def get_ocr_tesseract(img_path, labels_list):
    image = cv2.imread(img_path, 0)
    # thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dataset= {}
    for item in labels_list:
        label= item['label']
        x1= item['x1']
        y1= item['y1']
        x2= item['x2']
        y2= item['y2']
        ROI = image[y1:y2,x1:x2]
        labelled_text = pytesseract.image_to_string(ROI, lang='eng',config='--psm 6')
        # dataset[label] = []
        # dataset[label].append([labelled_text, [x1,y1, x2, y2]])
        if label in list(dataset.keys()):
            dataset[label].append([labelled_text, [x1,y1, x2, y2]])
        else:
            dataset[label] = []
            dataset[label].append([labelled_text, [x1,y1, x2, y2]])
    return dataset





        
        






def get_intersection_percentage(bb1, bb2):
    """
    Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
    """
    # print('entered into gip function ++++++++++++++====')
    # print(bb1['label'])
    # print(bb2['word'])
    # print(bb1['x1'])
    # print(bb1['x2'])
    # exit('')

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']


    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    #min_area = min(bb1_area,bb2_area)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if bb1_area>bb2_area:
        intersection_percent = intersection_area / bb2_area
    else:
        intersection_percent = intersection_area / bb1_area
        if intersection_percent<0.5:
            intersection_percent=1               # if ocr bounding box is big then we need to consider the entire token if the intersection is less than o.5 also 
            

    # print("The intersection percentage  of {text}, and {label},= {inter}".format(text = bb1['label'], label = bb2['word'], inter= intersection_percent))
    # exit('++++++++++++++==')
    assert intersection_percent >= 0.0
    assert intersection_percent <= 1.0
    return intersection_percent





# def get_intersection_percentage(bb1, bb2):
#     """
#     Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
#     """
#     print('entered into gip function ++++++++++++++====')
#     print(bb1['label'])
#     print(bb2['word'])
#     assert bb1['x1'] < bb1['x2']
#     assert bb1['y1'] < bb1['y2']
#     assert bb2['x1'] < bb2['x2']
#     assert bb2['y1'] < bb2['y2']

#     # determine the coordinates of the intersection rectangle
#     x_left = max(bb1['x1'], bb2['x1'])
#     y_top = max(bb1['y1'], bb2['y1'])
#     x_right = min(bb1['x2'], bb2['x2'])
#     y_bottom = min(bb1['y2'], bb2['y2'])

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0

#     # The intersection of two axis-aligned bounding boxes is always an
#     # axis-aligned bounding box
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     # compute the area of both AABBs
#     bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
#     bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
#     #min_area = min(bb1_area,bb2_area)
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     intersection_percent = intersection_area / bb2_area
#     print(f"The intersection percentage  of {bb1['label']}, and {bb2['word']},= {intersection_percent}")
#     # exit('++++++++++++++==')
#     assert intersection_percent >= 0.0
#     assert intersection_percent <= 1.0
#     return intersection_percent




if __name__== "__main__":

    #folder_path = "Data Generation"
    # folder_path = "Credit Note"
    # folder_path = "Credit Note updated"
    #folder_path = "Invoices combined"
    folder_path = "/home/ntlpt-42/Downloads/CS_ROOT_108"
    #pdf_path = os.path.join(folder_path, "1_page_pdfs")
    images_path = os.path.join(folder_path, "Images")
    labels_path = os.path.join(folder_path, "Labels")
    master_path = os.path.join(folder_path, "Master_Data")
    master_labels_path= os.path.join(folder_path, 'Master_Labels')
    ocr_path = os.path.join(folder_path, "OCR")
    #save_labels = os.path.join(folder_path, "labels_40_updated")
    if not os.path.exists(master_path):
        os.mkdir(master_path)
    if not os.path.exists(ocr_path):
        os.mkdir(ocr_path)
    if not os.path.exists(master_labels_path):
        os.mkdir(master_labels_path)
    # exit('++++++==')
    with open(os.path.join(folder_path, "label.txt"), "r") as f:
        classes = (f.read())
        classes = classes.split("\n")
    labelled_files = os.listdir(labels_path)
    labelled_files = [x.split(".txt")[0] for x in labelled_files]

    annotation_data = []
    thresh = 300
    count= 1
    for file in labelled_files:
        if os.path.exists(os.path.join(images_path, file + ".png")):
            print('yes')
            print(f'file name: {file}')
            # master_files = os.listdir(master_path)
            #doc = fitz.open(os.path.join(pdf_path, file + ".pdf"))
            image = cv2.imread(os.path.join(images_path, file + ".png"))
            word_coordinates = []
            word_coordinates, all_text = get_ocr_vision_api(os.path.join(images_path, file + ".png"))
            

            # exit('++++++')

            # print(word_coordinates)
            # exit('++++++++++==')

            # with open(os.path.join(ocr_path, file + "_text.txt"), "w") as f:
            #     json.dump({"word_coordinates": word_coordinates}, f)
            if len(word_coordinates) == 0:
                print(file)
                print("Not enough text")
            else:
                shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(master_path, file + ".png"))
                #shutil.copy(os.path.join(pdf_path, file + ".pdf"), os.path.join(master_path, file + ".pdf"))
                shutil.copy(os.path.join(labels_path, file + ".txt"), os.path.join(master_path, file + "_LabelImg.txt"))
                with open(os.path.join(master_path, file + "_text.txt"), "w") as f:
                    json.dump({"word_coordinates": word_coordinates}, f)
                with open(os.path.join(ocr_path, file + "_text.txt"), "w") as f:
                    json.dump({"word_coordinates": word_coordinates}, f)
                try:
                    with open(os.path.join(master_path, file + "_all_text.txt"), "w") as f:
                        json.dump({"all_text": all_text}, f)
                except:
                    pass
                
                h, w, _ = image.shape
                with open(os.path.join(labels_path, file + ".txt"), "r") as f:
                    label = (f.read())
                label = label.split("\n")
                labelled_data = []
                for l in label:
                    l = l.split()
                    if len(l) > 0:
                        l_class = classes[int(l[0])]
                        x_center = float(l[1]) * w
                        y_center = float(l[2]) * h
                        width = float(l[3]) * w
                        height = int(float(l[4]) * h)
                        x0 = int(x_center - (width/2))
                        x1 = int(x_center + (width/2))
                        y0 = int(y_center - (height / 2))
                        y1 = int(y_center + (height / 2))
                        cv2.rectangle(image, (x0, y0), (x1, y1), (0,255,0), 4)
                        labelled_data.append({
                            "label": l_class,
                            "x1": x0,
                            "y1": y0,
                            "x2": x1,
                            "y2": y1
                        })
                # print(labelled_data)
                # exit('+++++=')
                # cv2.imwrite(os.path.join("Labelling 2.png"), image)
                # break
                if len(labelled_data) > 0:
                    dataset = {}
                    for data in labelled_data:
                        overlapping_boxes = []
                        labelled_text = ""
                        for t in word_coordinates:
                            try:
                                print(data)
                                print(t)
                               
                                if get_intersection_percentage(data, t) >= 0.50:
                                    # exit('+++++++++')
                                    # print(get_intersection_percentage(data, t))
                                    t['label'] = data['label']
                                    overlapping_boxes.append(t)
                            except Exception as e:
                                print(t)
                                print(e)
                        # exit()
                        # overlapping_boxes = sorted(overlapping_boxes, key=cmp_to_key(contour_sort))
                        # print(data)
                        # print(overlapping_boxes)
                        for t in overlapping_boxes:
                            if len(labelled_text) == 0:
                                labelled_text = t['word']
                            else:
                                labelled_text += " " + t['word']
                        if len(labelled_text.strip()) == 0:
                            print(file + " - " + str(data) + " - " + str(len(overlapping_boxes)))
                        else:
                            if data['label'] in list(dataset.keys()):
                                dataset[data['label']].append([labelled_text, [data['x1'], data['y1'], data['x2'], data['y2']]])
                            else:
                                dataset[data['label']] = []
                                dataset[data['label']].append([labelled_text, [data['x1'], data['y1'], data['x2'], data['y2']]])
                    if dataset == {}:
                        print(file + " - blank")
                    remove_garbage(dataset)
                    with open(os.path.join(master_path, file + "_labels.txt"), "w") as f:
                        #f.write("\n,")
                        #f.write("\"File\": " + "\"" + file + "\",")
                        #f.write("\n")
                        json.dump(dataset, f)
                    with open(os.path.join(master_labels_path, file + "_labels.txt"), "w") as f:
                        #f.write("\n,")
                        #f.write("\"File\": " + "\"" + file + "\",")
                        #f.write("\n")
                        json.dump(dataset, f)
                    print(dataset)
                    # exit('+++++++++++++++++')
                
                        
        else:
            print("No")









