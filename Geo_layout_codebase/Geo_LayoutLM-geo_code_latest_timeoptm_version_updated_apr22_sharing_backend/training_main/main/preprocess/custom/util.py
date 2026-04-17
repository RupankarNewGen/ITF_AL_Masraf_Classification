import os
import shutil
from sklearn.model_selection import train_test_split
from typing import List
import json
import importlib.util
import sys

# Get the absolute path to constants.py (adjust the path as needed)
constants_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'constants.py'))
# Load the module dynamically
spec = importlib.util.spec_from_file_location("constants", constants_path)
constants = importlib.util.module_from_spec(spec)
sys.modules["constants"] = constants
spec.loader.exec_module(constants)

# Now you can access custom_split
CUSTOM_SPLIT = constants.custom_split
SPLIT_FILES = constants.spliting_files

def custom_splitting():
    train_image_files, val_image_files, train_annotation_files, val_annotation_files = [], [], [], []
    for split_mode, split_file in SPLIT_FILES.items():
        if split_file:
            with open(split_file, 'r') as file:
                img_files_list = [line.strip() for line in file.readlines()]
            annotation_json = [filename.replace('.jpeg', '.json') for filename in img_files_list]
            if split_mode == 'train':
                train_image_files = img_files_list
                train_annotation_files = annotation_json
            elif split_mode == 'test':
                val_image_files = img_files_list
                val_annotation_files = annotation_json
    return train_image_files, val_image_files, train_annotation_files, val_annotation_files



class DataSegmentation():
    def __init__(self, data_folder:str, train_folder:str, val_folder:str) -> None:
        THRESH= 200
        self.thresh= THRESH
        self.data_folder= data_folder
        self.train_folder= train_folder
        self.val_folder= val_folder
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        all_images= os.path.join(data_folder,'images')
        all_annotations= os.path.join(data_folder,'annotations')

        # List all files in the data folder
        all_images_lst: List = os.listdir(os.path.join(data_folder,'images'))
        all_annotations_lst : List= os.listdir(os.path.join(data_folder,'annotations'))

        # Separate image files and annotation files
        image_files = [file for file in all_images_lst if file.endswith('.jpg') or file.endswith('.jpeg')]
        annotation_files = [file for file in all_annotations_lst if file.endswith('.xml') or file.endswith('.json')]

        # Check if there are enough samples for splitting
        if len(image_files) < 2 or len(annotation_files) < 2:
            print("Insufficient data for splitting.")
        else:
            # Split image and annotation files into train and validation sets
            try:
                if CUSTOM_SPLIT:
                    train_image_files, val_image_files, train_annotation_files, val_annotation_files = custom_splitting()
                else:
                    train_image_files, val_image_files, train_annotation_files, val_annotation_files = train_test_split(
                        image_files, annotation_files, test_size=0.2, random_state=42
                    )
                # Create directories if they don't exist
                train_image= os.path.join(train_folder, 'images')
                train_annotation=  os.path.join(train_folder, 'annotations')
                os.makedirs(train_image, exist_ok=True)
                os.makedirs(train_annotation, exist_ok=True)

                test_image= os.path.join(val_folder, 'images')
                test_annotation= os.path.join(val_folder, 'annotations')
                os.makedirs(test_image, exist_ok=True)
                os.makedirs(test_annotation, exist_ok=True)
                # for image_file in train_image_files:
                #     image_file= image_file.split('.jpeg')[0]
                #     shutil.copy2(os.path.join(all_images, image_file+'.jpeg'), train_image)
                #     shutil.copy2(os.path.join(all_annotations, image_file+'.json'), train_annotation)
                # for image_file in val_image_files:
                #     image_file= image_file.split('.jpeg')[0]
                #     shutil.copy2(os.path.join(all_images, image_file+'.jpeg'), test_image)
                #     shutil.copy2(os.path.join(all_annotations, image_file+'.json'), test_annotation)
                
                for image_file in train_image_files:
                    image_file = image_file.split('.jpeg')[0]
                    source_image = os.path.join(all_images, image_file + '.jpeg')
                    source_annotation = os.path.join(all_annotations, image_file + '.json')

                    # Check if the image and annotation files exist before copying
                    if os.path.exists(source_image):
                        shutil.copy2(source_image, train_image)
                    else:
                        print(f"Warning: Image file '{source_image}' does not exist.")

                    if os.path.exists(source_annotation):
                        shutil.copy2(source_annotation, train_annotation)
                    else:
                        print(f"Warning: Annotation file '{source_annotation}' does not exist.")

                for image_file in val_image_files:
                    image_file = image_file.split('.jpeg')[0]
                    source_image = os.path.join(all_images, image_file + '.jpeg')
                    source_annotation = os.path.join(all_annotations, image_file + '.json')

                    # Check if the image and annotation files exist before copying
                    if os.path.exists(source_image):
                        shutil.copy2(source_image, test_image)
                    else:
                        print(f"Warning: Image file '{source_image}' does not exist.")

                    if os.path.exists(source_annotation):
                        shutil.copy2(source_annotation, test_annotation)
                    else:
                        print(f"Warning: Annotation file '{source_annotation}' does not exist.")
                print('Splitting has done successfully!')
                
            except Exception as e:
                print(e)
                exit("program exited due to mismatch size of image and labels")

    def __preprocess__(self, annot_path:str, file:str,data_folder:str, train_images_path:str):
        with open(os.path.join(annot_path,file+'.json')) as f:
            data= json.load(f)['form']
        if len(data)>self.thresh:
            print(file)
            print(len(data))
            # print(data)
            final_data=[]
            for i,item in enumerate(data):
                final_data.append(item)
                if  (i+1)%self.thresh==0:
                    file_name= file + "_s_" + str(int((i + 1) / self.thresh))
                    shutil.copy2(os.path.join(train_images_path,file+'.jpeg'),os.path.join(f'{data_folder}/Images',file_name+'.jpeg'))
                    print(final_data)
                    with open(os.path.join(os.path.join(f'{data_folder}/Annotations',file_name+'.json')), 'w') as f:
                        json.dump({"form" : final_data}, f, indent=4)
                    final_data.clear()
                if len(final_data)!=0:
                    file_name= file + "_s_" + str(int(len(data) /self.thresh) + 1)
                    shutil.copy2(os.path.join(train_images_path,file+'.jpeg'),os.path.join(f'{data_folder}/Images',file_name+'.jpeg')) 
                    with open(os.path.join(os.path.join(f'{data_folder}/Annotations',file_name+'.json')), 'w') as f:
                        json.dump({"form" : final_data}, f, indent=4)
        else:
            shutil.copy2(os.path.join(train_images_path,file+'.jpeg'), os.path.join(f'{data_folder}/Images', file+'.jpeg'))
            shutil.copy2(os.path.join(annot_path,file+'.json'), os.path.join(f'{data_folder}/Annotations', file+'.json'))

    def __trainDataSeg__(self):
        train_images_path= os.path.join(self.train_folder,'images')
        train_annot_path= os.path.join(self.train_folder,'annotations')
        image_files= os.listdir(train_images_path)
        image_files= [image.split('.jpeg')[0] for image in image_files]
        final_train_data_folder= os.path.join(self.data_folder, 'Train')
        final_image_path= os.path.join(final_train_data_folder, 'Images')
        final_labels_path= os.path.join(final_train_data_folder,'Annotations')
        os.makedirs(final_train_data_folder, exist_ok=True)
        os.makedirs(final_image_path, exist_ok=True)
        os.makedirs(final_labels_path, exist_ok=True)
        print(image_files)
        for file in image_files:
            if os.path.join(train_annot_path, file+'.json'):
                self.__preprocess__(train_annot_path,file,final_train_data_folder, train_images_path)
    def __testDataSeg__(self):
        test_images_path= os.path.join(self.val_folder,'images')
        test_annot_path= os.path.join(self.val_folder,'annotations')
        final_test_data_folder= os.path.join(self.data_folder, 'Test')
        final_image_path= os.path.join(final_test_data_folder, 'Images')
        final_labels_path= os.path.join(final_test_data_folder,'Annotations')
        os.makedirs(final_test_data_folder, exist_ok=True)
        os.makedirs(final_image_path, exist_ok=True)
        os.makedirs(final_labels_path, exist_ok=True)
        annot_files= os.listdir(test_annot_path)
        print(annot_files)
        image_files= os.listdir(test_images_path)
        image_files= [image.split('.jpeg')[0] for image in image_files]
        print(image_files)
        for file in image_files:
            if os.path.join(test_annot_path, file+'.json'):
                self.__preprocess__(test_annot_path, file, final_test_data_folder, test_images_path)
    


  

        













def Traintestsplit(data_folder:str, train_folder:str, val_folder:str):
    data_folder=data_folder
    train_folder=train_folder
    val_folder=val_folder
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    all_images= os.path.join(data_folder,'images')
    all_annotations= os.path.join(data_folder,'annotations')

    # List all files in the data folder
    all_images_lst: List = os.listdir(os.path.join(data_folder,'images'))
    all_annotations_lst : List= os.listdir(os.path.join(data_folder,'annotations'))

    # Separate image files and annotation files
    image_files = [file for file in all_images_lst if file.endswith('.jpg') or file.endswith('.jpeg')]
    annotation_files = [file for file in all_annotations_lst if file.endswith('.xml') or file.endswith('.json')]

    # print(image_files)
    # print(len(image_files))
    # # print(image_files[0])
    # exit('+++++++++++')

    # Check if there are enough samples for splitting
    if len(image_files) < 2 or len(annotation_files) < 2:
        print("Insufficient data for splitting.")
    else:
        # Split image and annotation files into train and validation sets\
        if CUSTOM_SPLIT:
            train_image_files, val_image_files, train_annotation_files, val_annotation_files = custom_splitting()

        else:
            train_image_files, val_image_files, train_annotation_files, val_annotation_files = train_test_split(
                image_files, annotation_files, test_size=0.2, random_state=42
            )
        # Create directories if they don't exist
        train_image= os.path.join(train_folder, 'images')
        train_annotation=  os.path.join(train_folder, 'annotations')
        os.makedirs(train_image, exist_ok=True)
        os.makedirs(train_annotation, exist_ok=True)

        test_image= os.path.join(val_folder, 'images')
        test_annotation= os.path.join(val_folder, 'annotations')
        os.makedirs(test_image, exist_ok=True)
        os.makedirs(test_annotation, exist_ok=True)



        # os.makedirs(train_annotation, exist_ok=True)
        # os.makedirs(test_annotation, exist_ok=True)
        # Copy image files to train and validation folders
        ##################################################################
        ##################################################################
        ##################################################################
        # for image_file in train_image_files:
        #     image_file= image_file.split('.jpeg')[0]
        #     shutil.copy2(os.path.join(all_images, image_file+'.jpeg'), train_image)
        #     shutil.copy2(os.path.join(all_annotations, image_file+'.json'), train_annotation)
        # for image_file in val_image_files:
        #     image_file= image_file.split('.jpeg')[0]
        #     shutil.copy2(os.path.join(all_images, image_file+'.jpeg'), test_image)
        #     shutil.copy2(os.path.join(all_annotations, image_file+'.json'), test_annotation)
            
        for image_file in train_image_files:
            image_file = image_file.split('.jpeg')[0]
            source_image = os.path.join(all_images, image_file + '.jpeg')
            source_annotation = os.path.join(all_annotations, image_file + '.json')

            # Check if the image and annotation files exist before copying
            if os.path.exists(source_image):
                shutil.copy2(source_image, train_image)
            else:
                print(f"Warning: Image file '{source_image}' does not exist.")

            if os.path.exists(source_annotation):
                shutil.copy2(source_annotation, train_annotation)
            else:
                print(f"Warning: Annotation file '{source_annotation}' does not exist.")

        for image_file in val_image_files:
            image_file = image_file.split('.jpeg')[0]
            source_image = os.path.join(all_images, image_file + '.jpeg')
            source_annotation = os.path.join(all_annotations, image_file + '.json')

            # Check if the image and annotation files exist before copying
            if os.path.exists(source_image):
                shutil.copy2(source_image, test_image)
            else:
                print(f"Warning: Image file '{source_image}' does not exist.")

            if os.path.exists(source_annotation):
                shutil.copy2(source_annotation, test_annotation)
            else:
                print(f"Warning: Annotation file '{source_annotation}' does not exist.")
                    
    print('Splitting has done successfully!')

if __name__== "__main__":
    data_folder = '/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/CS_complete_data/new_data_prep/data_in_funsd_format/test'
    # Paths to the train and validation folders
    train_folder= os.path.join(data_folder,'training_data')
    val_folder= os.path.join(data_folder,'testing_data')
    os.makedirs(os.path.join(data_folder,'training_data'),exist_ok=True)
    os.makedirs(os.path.join(data_folder,'testing_data'),exist_ok=True)

    data_seg= DataSegmentation(data_folder, train_folder, val_folder)

    data_seg.__testDataSeg__()
    data_seg.__trainDataSeg__()
 

