"""
Description
-----------

Perform OCR using AWS textract.
Update line no. 251 to update the Images path for which you want to 
apply the OCR operation.
"""


#Analyzes text in a document stored in an S3 bucket. Display polygon box around text and angled text 
import boto3
import json
import os
import copy
from tqdm import tqdm
from PIL import Image, ImageDraw

def ShowBoundingBox(draw,box,width,height,boxColor):
             
    left = width * box['Left']
    top = height * box['Top'] 
    draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline=boxColor)   
    return [left,top, left + (width * box['Width']), top +(height * box['Height'])]

def ShowSelectedElement(draw,box,width,height,boxColor):
             
    left = width * box['Left']
    top = height * box['Top'] 
    draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],fill=boxColor)  

# Displays information about a block returned by text detection and text analysis
def DisplayBlockInformation(block):
    print('Id: {}'.format(block['Id']))
    if 'Text' in block:
        print('    Detected: ' + block['Text'])
    print('    Type: ' + block['BlockType'])
   
    if 'Confidence' in block:
        print('    Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

    if block['BlockType'] == 'CELL':
        print("    Cell information")
        print("        Column:" + str(block['ColumnIndex']))
        print("        Row:" + str(block['RowIndex']))
        print("        Column Span:" + str(block['ColumnSpan']))
        print("        RowSpan:" + str(block['ColumnSpan']))    
    
    if 'Relationships' in block:
        print('    Relationships: {}'.format(block['Relationships']))
    print('    Geometry: ')
    print('        Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
    print('        Polygon: {}'.format(block['Geometry']['Polygon']))
    
    if block['BlockType'] == "KEY_VALUE_SET":
        print ('    Entity Type: ' + block['EntityTypes'][0])
    
    if block['BlockType'] == 'SELECTION_ELEMENT':
        print('    Selection element detected: ', end='')

        if block['SelectionStatus'] =='SELECTED':
            print('Selected')
        else:
            print('Not selected')    
    
    if 'Page' in block:
        print('Page: ' + block['Page'])
    print()

def process_text_analysis(client, document, image_name):

    
    # # Get the document from S3
    # try:                          
    #     with open(document, 'rb') as f:
    #     stream = f.read()
    # except Exception as e:
    #     print(f'The')
    try:
        with open(document,'rb') as f:
            stream= f.read()
    except Exception as e:
        print(e)
        print(f'The file {image_name} is not able to open in binary!')

    image_org=Image.open(document)
    image = Image.new('RGBA', image_org.size)
    image.paste(image_org)

    # Analyze the document
    image_binary = stream #.getvalue()
    try:
        print(type(image_binary))
        print(f'the image name in function: {document}')
        response = client.analyze_document(Document={'Bytes': image_binary},
            FeatureTypes=["TABLES", "FORMS", "SIGNATURES"])
    except Exception as e:
        print(e)
        print(f'The file {image_name} is not able run by the service')

    ### Uncomment to process using S3 object ###
    #response = client.analyze_document(
    #    Document={'S3Object': {'Bucket': bucket, 'Name': document}},
    #    FeatureTypes=["TABLES", "FORMS", "SIGNATURES"])

    ### Uncomment to analyze a local file ###
    # with open("pathToFile", 'rb') as img_file:
        ### To display image using PIL ###
    #    image = Image.open()
        ### Read bytes ###
    #    img_bytes = img_file.read()
    #    response = client.analyze_document(Document={'Bytes': img_bytes}, FeatureTypes=["TABLES", "FORMS", "SIGNATURES"])
    
    #Get the text blocks
    blocks=response['Blocks']
    width, height =image.size    
    # print ('Detected Document Text')
    
    # key_val_pairs = []
    # words_data = {}
    # values_data = {}
    # key_val_ids = [] 
    # words = []

    words_map = {}
    values_map = {}
    keys_map = {}

    # metadata = []
    # Create image showing bounding box/polygon the detected lines/text
    for i, block in enumerate(blocks):
        draw=ImageDraw.Draw(image)

        if block['BlockType'] == "WORD":
            coords = ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height,'white')
            coords = [int(item) for item in coords]
            words_map.update({block['Id']: {'text':block['Text'], 'bbox': coords}})
        # DisplayBlockInformation(block)    
        
        #print(block)
        # Draw bounding boxes for different detected response objects
        if block['BlockType'] == "KEY_VALUE_SET":
            #print(block)
            if block['EntityTypes'][0] == "KEY":
                # entity_type = 'key'
                coords = ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height,'red')
                values = []
                children = []

                for item in block['Relationships']:
                    if item['Type'] == 'VALUE':
                        values.extend(item['Ids'])
                    elif item['Type'] == 'CHILD':
                        children.extend(item['Ids'])

                keys_map.update({block['Id'] : {'bbox': coords, 'values': values, 'children': children}})



                
            else:
                # entity_type = 'val'
                coords = ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height,'green')  

                children = []
                if 'Relationships' in block:
                    # print(block['Relationships'])
                    for item in block['Relationships']:
                        if item['Type'] == 'CHILD':
                            children.extend(item['Ids'])
                    # children = [item['Ids'][0] for item in block['Relationships'] if item['Type'] == 'CHILD']

                values_map.update({block['Id'] : {'bbox': coords, 'children': children}})
                # values = []
            # if 'Relationships' in block:
            #     for rel in block['Relationships']:
            #         if rel['Type'] == 'CHILD':
            #             #key_val_ids.extend(rel['Ids'])
            #             if entity_type == 'val':
            #                 values_data.update({block['Id']: rel['Ids']})
            #             metadata.append({
            #                 'type': entity_type,
            #                 'bbox': [int(x) for x in coords],
            #                  'values': values,
            #                 'child_ids': rel['Ids']
            #             })

            
        if block['BlockType'] == 'TABLE':
            ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'blue')
        if block['BlockType'] == 'CELL':
            ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'yellow')
        if block['BlockType'] == 'SELECTION_ELEMENT':
            if block['SelectionStatus'] =='SELECTED':
                ShowSelectedElement(draw, block['Geometry']['BoundingBox'],width,height, 'blue')    
    
    final_data = []
    
    all_items = copy.copy(keys_map)
    all_items.update(values_map)
    all_ids = {}
    for i, key in enumerate(all_items): all_ids.update({key: i})
    
    for item in keys_map:
        
            key_bbox = keys_map[item]['bbox']
            key_text = []
            key_text_bbox = []

            value_text = []
            value_ids = []
            value_text_bbox=[]

            for child in keys_map[item]['children']:
                try:
                    key_text_bbox.append(words_map[child]['bbox'])
                    key_text.append(words_map[child]['text'])
                except:
                    pass

            
            for val in keys_map[item]['values']:
                value_bbox = values_map[val]['bbox']
                value_ids.append(val)
                for child in values_map[val]['children']:
                    try:
                        value_text.append(words_map[child]['text'])
                        value_text_bbox.append(words_map[child]['bbox'])
                    except:
                        pass
            
            if not len(key_text) < 1 and not len(value_text) < 1:

                final_data.append({
                    'key_text': key_text,
                    'value_text': value_text,
                    'key_id' : all_ids[item],
                    'value_ids' : [all_ids[item]  for item in value_ids],
                    'key_bbox': [int(key_coord) for key_coord in key_bbox],
                    'value_bbox': [int(val_coord) for val_coord in value_bbox],
                    'key_text_bbox': key_text_bbox,
                    'value_text_bbox': value_text_bbox
                })

    # for item in final_data:
    #     print(item)

    # exit()

    # image.show()
    # exit()
    # # Display the image
    #image.show()


    return len(blocks), final_data, words_map

def main():

    # session = boto3.Session(profile_name='profile-name')
    # s3_connection = session.resource('s3')
    client = boto3.client('textract', region_name='ap-south-1', aws_access_key_id='AKIA5QCZ7JK43AUK35DF',
                                   aws_secret_access_key='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',endpoint_url='https://textract.ap-south-1.amazonaws.com')

    

    
    root_path = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Layoutlmv3_code_base/BOL_complete_data/validation_data"
    img_path= os.path.join(root_path,'Images')
    images_list = os.listdir(img_path)

    save_to = os.path.join(root_path, 'custom_data/key_val_sets')
    save_words_to = os.path.join(root_path, 'custom_data/all_words')

    os.makedirs(save_words_to, exist_ok=True)
    os.makedirs(save_to, exist_ok=True)

    for image_name in tqdm(images_list, desc="Reading key-val pairs"):
        print(f'image name: {image_name}')
        document = os.path.join(img_path, image_name)
        block_count, key_val_pairs, all_words =process_text_analysis(client, document,image_name)
        # print("Blocks detected: " + str(block_count))
        
        json_name = image_name.replace('png', 'json')
        with open(os.path.join(save_to, json_name), 'w') as f:
            json.dump(key_val_pairs, f, indent=4)
        with open(os.path.join(save_words_to, json_name), 'w') as f:
            json.dump(all_words, f, indent=4)

if __name__ == "__main__":
    main()
