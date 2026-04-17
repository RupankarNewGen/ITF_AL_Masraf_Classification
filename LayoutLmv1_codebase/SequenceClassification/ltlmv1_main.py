"""
* *********************************************************************************
* Number Theory S/W Pvt. Ltd CONFIDENTIAL                                      *
* *
* [2016] - [2023] Number Theory S/W Pvt. Ltd Incorporated                       *
* All Rights Reserved.                                                          *
* *
* NOTICE:  All information contained herein is, and remains                     *
* the property of Number Theory S/W Pvt. Ltd Incorporated and its suppliers,    *
* if any.  The intellectual and technical concepts contained                    *
* herein are proprietary to Number Theory S/W Pvt. Ltd Incorporated             *
* and its suppliers and may be covered by India. and Foreign Patents,           *
* patents in process, and are protected by trade secret or copyright law.       *
* Dissemination of this information or reproduction of this material            *
* is strictly forbidden unless prior written permission is obtained             *
* from Number Theory S/W Pvt. Ltd Incorporated.                                 *
* *
* *********************************************************************************
"""

"""
Validates on
=============
1) ReviewID 1 = Hridesh Hari Dec12
"""


# import statements
##########################################################################################
from datetime import datetime
import os
import time
import numpy as np
import pandas as pd
from ocrpipeline import ApplyOcr
from torch.utils.data import random_split
from torch.optim import SGD, RMSprop
import torch
from PIL import Image
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array3D, Array2D
from transformers import AdamW, LayoutLMForSequenceClassification,LayoutLMTokenizer, LayoutLMConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import shutil
from seqeval.metrics import (
	classification_report)
from utility import get_logger_object_and_setting_the_loglevel, set_basic_config_for_logging
from tqdm import tqdm
#########################################################################################################


def convert_values(input_list, value_mapping):
    return [f'"{value_mapping.get(value, value)}' for value in input_list]

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]



def a_ocr(example):
    # print(ocr_path)
    name = example['image_path'].split("/")[-1].split(".")[0]
    path = os.path.join(ocr_path,name+".json")
    
    words,boxes = ApplyOcr.apply_ocr_gv(example,ocr_path)
    # try:
    #     data0 = eval(j_data)
    # except:
    #     data0  = j_data
    # print(data0)
    # dic = data0
    # keys = dic.keys()
    # words = []
    # boxes = []
    # for key in keys:
    #     words.append(key)
    #     boxes.append(dic[key])

    # add as extra columns
    assert len(words) == len(boxes)
    example['words'] = words
    example['bbox'] = boxes
    return example


def encode_example_1(example):
    images = [Image.open(path).convert("RGB") for path in example['image_path']]
    encoding = processor(images, example['words'], boxes=example['bbox'], padding="max_length", truncation=True)#, return_tensors="pt")#, truncation=True)#, apply_ocr=False)
    encoding["labels"] = [label2idx[label] for label in example["label"]]
    encoding['image_path'] = example['image_path']
    return encoding

def encode_example(example, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):
    # print("example")
    # print(example["image_path"])
    words = example['words']
    normalized_word_boxes = example['bbox']
    assert len(words) == len(normalized_word_boxes)

    token_boxes = []
    for word, box in zip(words, normalized_word_boxes):
        word_tokens = processor.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))
    # Truncation of token_boxes
    special_tokens_count = 2 
    if len(token_boxes) > max_seq_length - special_tokens_count:
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

    # add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    encoding = processor(' '.join(words), padding='max_length', truncation=True)
    # Padding of token_boxes up the bounding boxes to the sequence length.
    input_ids = processor(' '.join(words), truncation=True)["input_ids"]
    padding_length = max_seq_length - len(input_ids)
    token_boxes += [pad_token_box] * padding_length
    encoding['bbox'] = token_boxes
    encoding['label'] = label2idx[example['label']]
    assert len(encoding['input_ids']) == max_seq_length
    assert len(encoding['attention_mask']) == max_seq_length
    assert len(encoding['token_type_ids']) == max_seq_length
    assert len(encoding['bbox']) == max_seq_length
    # print("EXAMPLE DONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
    return encoding

def modify_llm_config(o_config):
    # Create a new configuration object based on the original configuration
    m_config = LayoutLMConfig.from_dict(o_config.to_dict())

    # Modify the desired parameters
    m_config.num_labels = len(label2idx)
    m_config.vocab_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["vocabSize"])
    m_config.hidden_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["hiddenSize"])
    m_config.num_hidden_layers = int(input_json_dict["layoutLMv2"]["modelConfig"]["numHiddenLayers"])
    m_config.num_attention_heads = int(input_json_dict["layoutLMv2"]["modelConfig"]["numAttentionHeads"])
    m_config.intermediate_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["intermediateSize"])
    m_config.hidden_act = str(input_json_dict["layoutLMv2"]["modelConfig"]["hiddenAct"])
    m_config.hidden_dropout_prob = float(input_json_dict["layoutLMv2"]["modelConfig"]["hiddenDropoutProb"])
    m_config.attention_probs_dropout_prob = float(
        input_json_dict["layoutLMv2"]["modelConfig"]["attentionProbsDropoutProb"])
    m_config.max_position_embeddings = int(input_json_dict["layoutLMv2"]["modelConfig"]["maxPositionEmbeddings"])
    m_config.type_vocab_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["typeVocabSize"])
    m_config.initializer_range = float(input_json_dict["layoutLMv2"]["modelConfig"]["initializerRange"])
    m_config.layer_norm_eps = float(input_json_dict["layoutLMv2"]["modelConfig"]["layerNormEps"])
    m_config.pad_token_id = int(input_json_dict["layoutLMv2"]["modelConfig"]["padTokenId"])
    m_config.max_2d_position_embeddings = int(input_json_dict["layoutLMv2"]["modelConfig"]["max2dPositionEmbeddings"])
    m_config.max_rel_pos = int(input_json_dict["layoutLMv2"]["modelConfig"]["maxRelPos"])
    m_config.rel_pos_bins = int(input_json_dict["layoutLMv2"]["modelConfig"]["relPosBins"])
    m_config.fast_qkv = bool(input_json_dict["layoutLMv2"]["modelConfig"]["fastQkv"])
    m_config.max_rel_2d_pos = int(input_json_dict["layoutLMv2"]["modelConfig"]["maxRel2dPos"])
    m_config.rel_2d_pos_bins = int(input_json_dict["layoutLMv2"]["modelConfig"]["rel2dPosBins"])
    m_config.convert_sync_batchnorm = bool(input_json_dict["layoutLMv2"]["modelConfig"]["convertSyncBatchnorm"])
    m_config.image_feature_pool_shape = [int(val) for val in
                                         input_json_dict["layoutLMv2"]["modelConfig"]["imageFeaturePoolShape"]]
    m_config.coordinate_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["coordinateSize"])
    m_config.shape_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["shapeSize"])
    m_config.has_relative_attention_bias = bool(
        input_json_dict["layoutLMv2"]["modelConfig"]["hasRelativeAttentionBias"])
    m_config.has_spatial_attention_bias = bool(input_json_dict["layoutLMv2"]["modelConfig"]["hasSpatialAttentionBias"])
    m_config.has_visual_segment_embedding = bool(
        input_json_dict["layoutLMv2"]["modelConfig"]["hasVisualSegmentEmbedding"])
    # m_config.detectron2_config_args = dict(input_json_dict["layoutLMv2"]["modelConfig"]["detectron2ConfigArgs"])
    return m_config


def execute_optimizer_process(optimizer, model, lr, input_json_dict):
    switch = {
        "adam": lambda: AdamW(model.parameters(), lr),
        "sgd": lambda: SGD(model.parameters(), lr,
                           momentum=float(input_json_dict["layoutLMv2"]["generalConfig"]["momentum"])),
        "rmsprop": lambda: RMSprop(model.parameters(), lr,
                                   alpha=float(input_json_dict["layoutLMv2"]["generalConfig"]["alpha"]))
    }
    return switch.get(optimizer, lambda: print("Invalid optimizer"))()


def metric_calculation(u_true, u_pred):
    # Convert true labels and predicted labels to numpy arrays
    true_labels = np.array(u_true)
    predicted_labels = np.array(u_pred)

    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            "Inconsistent numbers of samples: true_labels: {}, predicted_labels: {}".format(len(true_labels),
                                                                                            len(predicted_labels)))
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(true_labels, predicted_labels, average="weighted")
    print("Precision:", precision)

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels, average="weighted")
    print("Recall:", recall)

    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    print("F1 Score:", f1)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(confusion_mat)
    return accuracy, precision, recall, f1, confusion_mat


def filter_updated_data(updated_dataset, new_updated_dataset):
    for i in range(len(updated_dataset)):
        if len(updated_dataset[i]['words']) !=0 and len(updated_dataset[i]['bbox']) !=0:
            updated_dataset_len = len(updated_dataset[i]['words'])
            if updated_dataset_len > 250:
                my_list_words = updated_dataset[i]['words']
                my_list_bbox = updated_dataset[i]['bbox']
                chunk_size = 250
                word_chunks = []
                bbox_chunks = []
                for k in range(0, len(my_list_words), chunk_size):
                    chunk = my_list_words[k:k + chunk_size]
                    word_chunks.append(chunk)
                    chunk = my_list_bbox[k:k + chunk_size]
                    bbox_chunks.append(chunk)
                    if len(word_chunks[-1]) != len(bbox_chunks[-1]):
                        raise Exception("length Mismatch")
                # shutil.copy(updated_dataset[i]['image_path'], updated_dataset[i]['image_path'][0:-4]+'1'+'.png')
                # print(updated_dataset[i]['image_path'][0:-4]+'1'+'.png')
                # exit("********")
                for j in range(len(word_chunks)):
                    new_row = {'image_path': updated_dataset[i]['image_path'], 'label': updated_dataset[i]['label'], 'words': word_chunks[j], 'bbox': bbox_chunks[j]}
                    new_row_df = pd.DataFrame([new_row])
                    new_updated_dataset = pd.concat([new_updated_dataset, new_row_df], ignore_index=True)
            else:
                new_row = {'image_path': updated_dataset[i]['image_path'], 'label': updated_dataset[i]['label'], 'words': updated_dataset[i]['words'], 'bbox': updated_dataset[i]['bbox']}
                new_row_df = pd.DataFrame([new_row])
                new_updated_dataset = pd.concat([new_updated_dataset, new_row_df], ignore_index=True)
    return new_updated_dataset

def filter_empty_words_cords(df,c1,c2):
    df_filtered = df[((df[c1]!= '[]') & (df[c2] != '[]'))]
    return df_filtered

def train_test_split_base(dataset, flag):
    if flag=='train':
        dataset = Dataset.from_dict(dataset)
        updated_dataset = dataset.map(a_ocr)
        columns_names = ['image_path', 'label', 'words', 'bbox']  # Replace with your desired column names
        new_updated_dataset = pd.DataFrame(columns=columns_names)
        new_updated_dataset = filter_updated_data(updated_dataset, new_updated_dataset)
        # new_updated_dataset.to_csv("test3.csv",index=False)
        # new_updated_dataset = pd.read_csv("/home/ng6309/datascience/hridesh/layoutlmv1/test3.csv")
        new_updated_dataset = Dataset.from_pandas(new_updated_dataset)
        # encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example), features=training_features, batched=False, batch_size=1)
        encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example))
        encoded_dataset.set_format(type='torch', device=device)
        return encoded_dataset
    elif flag=='test':
        dataset = Dataset.from_dict(dataset)
        updated_dataset = dataset.map(a_ocr)
        columns_names = ['image_path', 'label', 'words', 'bbox']  # Replace with your desired column names
        new_updated_dataset = pd.DataFrame(columns=columns_names)
        new_updated_dataset = filter_updated_data(updated_dataset, new_updated_dataset)
        new_updated_dataset = Dataset.from_pandas(new_updated_dataset)
        # encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example), features=training_features, batched=False, batch_size=1)
        encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example))
        encoded_dataset.set_format(type='torch', device=device)
        return encoded_dataset    
        

################################################################################################################################

"""________________________________________MAIN STARTS HERE______________________________________________________"""

input_json_dict = {"layoutLMv2": {
    "modelConfig": {
        "vocabSize": "30522",
        "hiddenSize": "768",
        "numHiddenLayers": "12",
        "numAttentionHeads": "12",
        "intermediateSize": "3072",
        "hiddenAct": "gelu",
        "hiddenDropoutProb": "0.1",
        "attentionProbsDropoutProb": "0.1",
        "maxPositionEmbeddings": "512",
        "typeVocabSize": "2",
        "initializerRange": "0.02",
        "layerNormEps": "1e-12",
        "padTokenId": "0",
        "max2dPositionEmbeddings": "1024",
        "maxRelPos": "128",
        "relPosBins": "32",
        "fastQkv": "True",
        "maxRel2dPos": "256",
        "rel2dPosBins": "64",
        "convertSyncBatchnorm": "True",
        "imageFeaturePoolShape": [7, 7, 256],
        "coordinateSize": "128",
        "shapeSize": "128",
        "hasRelativeAttentionBias": "True",
        "hasSpatialAttentionBias": "True",
        "hasVisualSegmentEmbedding": "False",
        # "detectron2ConfigArgs": {}
    },
    "generalConfig": {
        "epoch": "40",
        "batchSize": "4",
        "shuffle": "True",
        "learningRate": "1e-5",
        "optimizer": "adam",
        "momentum": "0.9",
        "alpha": "0.9"
    },
    "device": "cuda"
}}


# Main Code
#####################################################################################
# Device: Linux
# Configuration: 16G
#####################################################################################
set_basic_config_for_logging(filename=f"logs/classification_training_{datetime.now()}")
logger = get_logger_object_and_setting_the_loglevel()

global ocr_path

dataset_path = "/home/ng6309/datascience/hridesh/layoutlmv1/classification/BILLS"
ocr_path = '/home/ng6309/datascience/hridesh/OCR_Dec12'

logger.info("Starting training")

labels = list(os.listdir(dataset_path))
idx2label = dict(enumerate(labels))
label2idx = {k: v for v, k in enumerate(labels)}



training_features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    # 'labels': ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
    'image_path': Value(dtype='string'),
    'label': Value(dtype='int32'),
    'words': Sequence(feature=Value(dtype='string'))
})

logger.info(f"Labels: {labels}")
logger.info(f"idx2label: {idx2label}")
logger.info(f"label2idx: {label2idx}")


with open('label.txt', 'w') as label_file:
	label_file.write(json.dumps(label2idx))

images = []
labels = []
for label in os.listdir(dataset_path):
    images.extend([
        f"{dataset_path}/{label}/{img_name}" for img_name in os.listdir(f"{dataset_path}/{label}")
    ])
    labels.extend([
        label for _ in range(len(os.listdir(f"{dataset_path}/{label}")))
    ])

data = pd.DataFrame({'image_path': images, 'label': labels})
data.to_csv("images_and_labels.csv")    


logger.info("images and labels reading and dataframe creation done")

processor = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

device = torch.device(input_json_dict["layoutLMv2"]["device"] if torch.cuda.is_available() else "cpu")
label_list = data["label"].tolist()
print(data.columns)
datasetall = Dataset.from_pandas(data)

# training_set, testing_set = train_test_split(datasetall, test_size=0.1, random_state=42)
training_set, testing_set = train_test_split(data, test_size=0.2, random_state=42,stratify=label_list)
# validation_set, testing_set = train_test_split(data, test_size=0.5, random_state=42)

print(datasetall.column_names)


df1 = pd.DataFrame(training_set)
df2 = pd.DataFrame(testing_set)
# df3 = pd.DataFrame(validation_set)
# Write the DataFrames to separate sheets in the Excel file
df1.to_csv('training_set_1.csv')
df2.to_csv('testing_set_1.csv')
# df3.to_csv('validation_set.csv')


print("======================Date count training data==================")
print(df1["label"].value_counts())
logger.info("================================label count for train  data =================")
logger.info(df1["label"].value_counts())
print("======================Date count testing data=====================")
print(df2["label"].value_counts())
logger.info("================================label count for test data =================")
logger.info(df2["label"].value_counts())


train_data = train_test_split_base(training_set , 'train')
valid_data = train_test_split_base(testing_set , 'test')
# train_data = train_data.remove_columns(["words", "label"])
# valid_data = valid_data.remove_columns(["words", "label"])

print(train_data["label"])
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=True)

'''------------------------------DEFINING THE MODEL---------------------------------'''
original_config = LayoutLMConfig.from_pretrained("microsoft/layoutlm-base-uncased")
# Modify the configuration as needed
modified_config = modify_llm_config(original_config)
# Add more modifications as needed
# model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased",
#                                                             config=modified_config)


model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased",num_labels=len(label2idx))
model.to(device)
'''------------------------------TRAINING THE MODEL---------------------------------'''
best_model = None
best_metric = float('-inf')
best_epoch = 0
best_time = float('inf')
best_loss = None#float('inf')
best_performance = 0.0
lr = float(input_json_dict["layoutLMv2"]["generalConfig"]["learningRate"])
optimizer_str = str(input_json_dict["layoutLMv2"]["generalConfig"]["optimizer"])
optimizer = execute_optimizer_process(optimizer_str, model, lr, input_json_dict)
global_step = 0
num_epochs = int(input_json_dict["layoutLMv2"]["generalConfig"]["epoch"])
start_time = time.time()

# put the model in training mode
for epoch in range(num_epochs):
    print("Epoch:", epoch)
    # for training data
    training_loss = 0.0
    training_correct = 0
    # put the model in training mode
    model.train()

    for batch in tqdm(train_dataloader):
        # labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = torch.tensor(batch["label"])
        # outputs = model(
        #     image=batch["image"].to(device),
        #     input_ids=batch["input_ids"].to(device), bbox=batch["bbox"].to(device),
        #     attention_mask=batch["attention_mask"].to(device),
        #     token_type_ids=batch["token_type_ids"].to(device),
        #     labels=labels
        # )
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                      labels=labels)
        # forward pass
        loss = outputs.loss

        training_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        training_correct += (predictions == labels).float().sum()
        # backward pass
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    with open("final_result.txt", 'a') as f:
        print("Training Loss:", training_loss / len(train_data))
        f.write(f"Training Loss: :{training_loss / len(train_data)}\n")
        f.write(f"loss value after complete epoch : {loss}")
        training_accuracy = 100 * training_correct / len(train_data)
        print("Training accuracy:", training_accuracy.item())
        f.write(f"training_accuracy :{training_accuracy.item()}\n")


    predicted_labels = []
    true_labels = []
    # For validation data
    validation_loss = 0.0
    validation_correct = 0
    for batch2 in valid_dataloader:
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = torch.tensor(batch["label"])
        t = labels.tolist()
        true_labels.append(t)
        # outputs = model(
        #     image=batch2["image"].to(device),
        #     input_ids=batch2["input_ids"].to(device), bbox=batch2["bbox"].to(device),
        #     attention_mask=batch2["attention_mask"].to(device),
        #     token_type_ids=batch2["token_type_ids"].to(device),
        #     labels=labels
        # )
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                      labels=labels)
        # forward pass
        loss = outputs.loss
        validation_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        validation_correct += (predictions == labels).float().sum()
        x = predictions.tolist()
        predicted_labels.append(x)

    with open("final_result.txt", 'a') as f:
        print("Validation Loss:", validation_loss / len(valid_data))
        f.write(f"Validation Loss: :{validation_loss / len(valid_data)}\n")
        validation_accuracy = 100 * validation_correct / len(valid_data)
        print("Validation accuracy:", validation_accuracy.item())
        f.write(f"Validation accuracy: {validation_accuracy.item()}\n")
        validation_loss = validation_loss / len(valid_data)

    if best_loss is None:
        best_loss = validation_loss

    if validation_loss <= best_loss:
        best_metric = validation_accuracy
        best_model = model
        best_loss = validation_loss
        best_epoch = epoch
        best_time = time.time() - start_time
        best_performance = validation_accuracy


    unlisted_true = [item[0] for item in true_labels]
    print(unlisted_true)
    unlisted_predicted = [item[0] for item in predicted_labels]
    print(unlisted_predicted)
    #model.save_pretrained(f'saved_model_{epoch}')
    accuracy, precision, recall, f1, confusion_mat = metric_calculation(unlisted_true, unlisted_predicted)

    ground_truth = convert_values(unlisted_true, idx2label)
    pred_val = convert_values(unlisted_predicted, idx2label)

    with open("final_result.txt", 'a') as f:
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"f1: {f1}\n")
        f.write(f"{confusion_mat}\n")
        f.write(classification_report([ground_truth], [pred_val]))    

# to calculate metrics for the best model
torch.save(best_model, 'best_model.pt')
best_model.save_pretrained('best_model_pre')
print(f"Best Model: Epoch {best_epoch}, "
      f"Time: {best_time:.2f} seconds, "
      f"Loss: {best_loss:.4f}, "
      f"Performance: {best_performance:.4f}")
with open("final_result.txt", 'a') as f:
    f.write(f"Best Model: Epoch {best_epoch}, "
      f"Time: {best_time:.2f} seconds, "
      f"Loss: {best_loss:.4f}, "
      f"Performance: {best_performance:.4f}\n")

"""_______________________________________________METRICS______________________________________________________"""

print(" CODE EXECUTED SUCCESSFULLY___________________________!!!!!!!!!!!!!!!!!!!!")

