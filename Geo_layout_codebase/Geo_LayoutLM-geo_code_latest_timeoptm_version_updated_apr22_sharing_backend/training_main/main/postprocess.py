import cv2
import os 
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
from bounding_box_using_pred_key import draw_rectangles
# from draw_prediction import draw_rectangles

def contour_sort(a, b):
	if abs(a['y1'] - b['y1']) <= 15:
		return a['x1'] - b['x1']

	return a['y1'] - b['y1']
root_path= "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Layoutlmv3_code_base/test_code/Validation_data"
res_path = '/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Layoutlmv3_code_base/test_code/Validation_data/custom_results/custom_trial'
image_path = os.path.join(root_path, 'data_in_funsd_format/dataset/custom_geo/testing_data/images')
act_path = os.path.join(root_path,'data_in_funsd_format/testing_data/annotations/')
print(act_path)
# exit('++++++++++++++++++=')
output_image_path= os.path.join(root_path, "res_viz_test")

if not os.path.exists(output_image_path):
    os.mkdir(output_image_path)

results = [item for item in os.listdir(res_path) if item.endswith('json')]

# print(results)
# print(len(results))

# print(output_image_path)
# print(image_path)

# exit('+++++++++++')


key_map = {}
for res in tqdm(results):
    with open(os.path.join(res_path, res), 'r') as f:
        data = json.load(f)

    act_res = res.replace('_tagging.json', '.json')
    with open(os.path.join(act_path, act_res), 'r') as f:
        actual_data = json.load(f)['form']

    actual_map = {item['label']: ''.join(item['text'].lower().split()) for item in actual_data if not item['label'] == "other"}
    actual_map_coords = {item['label']: item['box'] for item in actual_data if not item['label'] == "other"}

    # print(actual_map)
    print(f'file name: {res}')
    print('++++++++++++++++=== Acutal data ++++++======')
    print(actual_data)
    print('++++++++++++++++= pred data ++++++++++++++++++++++++++')
    print(data)

    exit('++++++++++++')

    key_coords = {}

    #for item in actual_map:
    #    print(item, '  :::  ', actual_map[item])

    #break

    unique_keys = [item['pred_key'][2:] for item in data]
    
    fin = {}
    for item in data:
        if item['pred_key'][2:] != '':
            if item['pred_key'][2:] not in fin and 'SEP' not in item['text']:
                fin[item['pred_key'][2:]] = item['text']
                key_coords[item['pred_key'][2:]] = item['coords']
            elif item['pred_key'][2:] in fin and 'SEP' not in item['text']:
                fin[item['pred_key'][2:]] += f"{item['text']}"
                key_coords[item['pred_key'][2:]] = [key_coords[item['pred_key'][2:]][0], key_coords[item['pred_key'][2:]][1], item['coords'][2], item['coords'][3]]
            #key_coods.append()
           # print(item['pred_key'][2:],fin[item['pred_key'][2:]])

    #for item in fin:
    #    print(item, ' ::: ', fin[item])
    #print(unique_keys)
    
    image_name = res.replace('_tagging.json','.png')
    draw_rectangles(data, os.path.join(image_path, image_name), output_image_path,image_name )

    

    comp = []

    image = cv2.imread(os.path.join(image_path, image_name))

    for item in actual_map:
        cv2.rectangle(image, (actual_map_coords[item][0], actual_map_coords[item][1]), (actual_map_coords[item][2], actual_map_coords[item][3]), (0, 0, 255), 4)
        cv2.putText(image, item, (actual_map_coords[item][0], actual_map_coords[item][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        complete_match = 0
        if item in fin:
            text = ''.join(fin[item].split())
            cv2.rectangle(image, (key_coords[item][0], key_coords[item][1]), (key_coords[item][2], key_coords[item][3]), (255, 0, 0), 4)
            cv2.rectangle(image, (key_coords[item][0], key_coords[item][1]), (key_coords[item][2], key_coords[item][3]), (255, 0, 0), 4)
            cv2.putText(image, item, (key_coords[item][0], key_coords[item][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        else:
            text = ''

        #print(text, actual_map[item])
        #exit()
        if text == actual_map[item]:
        #if list(set(fin[item]) - set(actual_map[item])) == []:
            complete_match = 1
        comp.append({
            'key': item,
            'actual': actual_map[item],
            'pred': text,
            'complete_match': complete_match
           })

       
        cv2.imwrite(os.path.join('res_viz', image_name), image)
        
        if item in key_map:
            key_map[item]['complete_match'] += complete_match
            key_map[item]['total'] += 1
        elif item not in key_map:
            key_map.update({item : {'complete_match':complete_match,'total':1, 'key_accuracy': 0}})
    """
    #print("Manual validation reqiured for the following: ")
    for item in fin:
        if item not in actual_map:
            print('*'*20)
            print(f"File name: {res}")
            print(f"Key :::::::  {item}")
            #print(f"Coords ::::::: {fin}")
            print(f"Text :::::: {fin[item]}")
            print('*'*20, end='\n\n')
    """



#for item in key_map:
#    print(item , ' :: ', key_map[item])


csv_res = {'key': [], 'complete_match': [], 'total':[], 'key_accuracy': []}
for item in key_map:
    #print(key_map[item]['key_accuracy'])
    key_map[item]['key_accuracy'] = round((key_map[item]['complete_match']/key_map[item]['total']) * 100, 2)

    csv_res['key'].append(item)
    csv_res['complete_match'].append(key_map[item]['complete_match'])
    csv_res['total'].append(key_map[item]['total'])
    csv_res['key_accuracy'].append(key_map[item]['key_accuracy'])

df = pd.DataFrame(csv_res)
df.to_csv(os.path.join(root_path,'final_result_test.csv'), index=None)

with open(os.path.join(root_path, 'final_result_test.json'), 'w') as f:
    json.dump(key_map, f, indent=4)

    #text = sorted(text, key=cmp_to_key(contour_sort))  


