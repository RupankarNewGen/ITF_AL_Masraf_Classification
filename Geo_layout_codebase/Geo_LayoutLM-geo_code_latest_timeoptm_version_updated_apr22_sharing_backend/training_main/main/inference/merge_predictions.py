import os
import json
import shutil
def main():
    input_path= "/home/gpu1admin/rakesh/geo_testing/data_oct21/data_in_funsd_format/dataset/custom_trial"
    images_path= "/home/gpu1admin/rakesh/geo_testing/data_oct21/Images"
    anno_dir= 'annotations'
    images_dir= 'images'
    result_path_anno = os.path.join(input_path, f"Results/{anno_dir}")
    os.makedirs(result_path_anno,exist_ok=True)
    # if not os.path.exists(result_path_anno):
    #     os.mkdir(result_path_anno)
    result_path_img= os.path.join(input_path,f"Results/{images_dir}")
    if not os.path.exists(result_path_img):
        os.mkdir(result_path_img)
    # img_path= os.path.join(input_path, "vis")
    image_list= os.listdir(input_path)
    image_list=  [image.split('_tagging.json')[0] for image in image_list if image.endswith(".json")]
    image_count={}
    for file in image_list:
        if len(file.split('_s_'))>1:
            key= file.split('_s_')[0]
            value= file.split('_s_')[1]
            if key not in image_count:
                image_count[key]= int(value)
            else:
                if int(image_count[key])< int(value):
                    image_count[key]= int(value)
        else:
            image_count[file]=1
    print(image_count)
    # exit('++++++++++++++++')
    # print(image_count)
    for file in list(image_count.keys()):
        print(f'processing:{file}')
        print(file)
        if int(image_count[file])>1:
            final_dict=[]
            for i in range(0, int(image_count[file])):
                file_name= f"{file}_s_"+str(i+1)
                if os.path.exists((os.path.join(input_path, file_name+'_tagging.json'))):
                    with open(os.path.join(input_path, file_name+'_tagging.json'), 'r') as f:
                        data= json.load(f)
                    for item in data:
                        # print(f'item value: {item}')
                        if "[SEP]" in item['text']:
                            item['text']=item['text'].replace("[SEP]", "")  #removing [SEP] from the json 
                        final_dict.append(item)
            with open(os.path.join(result_path_anno,file+'_tagging.json'), 'w') as f:
                json.dump({"form" : final_dict}, f, indent=4)
            try:
                shutil.copy2(os.path.join(images_path,file+'.png'), os.path.join(result_path_img,file+'_linking.png'))
            except Exception as e:
                continue
        else:
            try:
                shutil.copy2(os.path.join(images_path,file+'.png'), os.path.join(result_path_img,file+'_linking.png'))
            except Exception as e:
                continue
            shutil.copy2(os.path.join(input_path,file+'_tagging.json'), os.path.join(result_path_anno,file+'_tagging.json'))
if __name__=='__main__':
    main()
