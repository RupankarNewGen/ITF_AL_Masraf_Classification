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

processor = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

def encode_example(example, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):
    # print("example")
    label2idx = {"PO": 0, "PI": 1, "OTHERS": 2}
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
    encoding['label'] = label2idx[example['label']]
    encoding['bbox'] = token_boxes
    assert len(encoding['input_ids']) == max_seq_length
    assert len(encoding['attention_mask']) == max_seq_length
    assert len(encoding['token_type_ids']) == max_seq_length
    assert len(encoding['bbox']) == max_seq_length
    # print("EXAMPLE DONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
    return encoding

ocr_path="/home/ng6309/datascience/hridesh/layoutlmv1/classification/EVL_OCR"

def apply_ocr(example):
    # print(ocr_path)
    name = example['image_path'].split("/")[-1].split(".")[0]
    path = os.path.join(ocr_path,name+".json")
    
    words,boxes = ApplyOcr.apply_ocr_gv(example,ocr_path)

    assert len(words) == len(boxes)
    example['words'] = words
    example['bbox'] = boxes
    return example

def update_dataframe(df,imagename, value_list):
    # Check if the imagename already exists in the DataFrame
    if imagename in df['imagename'].values:
        # Append the new values to the existing list
        df.loc[df['imagename'] == imagename, 'values'].iloc[0].extend([value_list])
    else:
        # Add a new row with the imagename and the list of values
        df = pd.concat([df, pd.DataFrame({'imagename': [imagename], 'values': [[value_list]]})], ignore_index=True)
    return df

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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LayoutLMForSequenceClassification.from_pretrained("/home/ng6309/datascience/hridesh/layoutlmv1/classification/best_model_pre")
model.to(device)

query_df = pd.DataFrame(
    {'image_path': ['/home/ng6309/datascience/hridesh/layoutlmv1/classification/EVL/LC/PO/Purchase_Order_By_Opener_190_1.png'],
    "label":["PO"]}
)
query = Dataset.from_pandas(query_df)
query = query.map(apply_ocr)
columns_names = ['image_path', 'label', 'words', 'bbox']
new_updated_dataset = pd.DataFrame(columns=columns_names)
new_updated_dataset = filter_updated_data(query, new_updated_dataset)
new_updated_dataset = Dataset.from_pandas(new_updated_dataset)
query = new_updated_dataset
query = query.map(lambda example: encode_example(example))
# query.set_format(
#     type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids']
# )

query.set_format(type='torch')
query = torch.utils.data.DataLoader(query)
# batch = next(iter(query))

df = pd.DataFrame(columns=['imagename', 'values'])

for batch in query:
    outputs = model(
        input_ids=batch["input_ids"].to(device), bbox=batch["bbox"].to(device), 
        attention_mask=batch["attention_mask"].to(device), 
        token_type_ids=batch["token_type_ids"].to(device)
    )
    label2idx = {"PO": 0, "PI": 1, "OTHERS": 2}
    preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
    print(preds)
    pred_labels = {label:pred for label, pred in zip(label2idx.keys(), preds)}
    image_name = batch["image_path"][0]
    df = update_dataframe(df,image_name, preds)

print(df)

new_vals = []

for  i in df["values"].tolist():
    print(i)

df["pred"] = df["values"].apply(lambda x:np.mean(np.array(x), axis=0))

print(df["pred"])