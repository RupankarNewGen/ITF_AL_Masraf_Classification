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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

processor = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")


label_path = '/home/ng6309/datascience/hridesh/layoutlmv1/classification/label.txt'

global labeltoidx
global label2idx
global idx2label

with open(label_path, "r") as f:
    label_train_gen = f.read().splitlines()

label_dict = json.loads(label_train_gen[0])
print(label_dict)
labeltoidx = label_dict
label2idx = labeltoidx

idx2label = {v: k for v, k in enumerate(label2idx)}
# exit()

label_list = list(labeltoidx.keys())
classes = label_list
print(classes)


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
    encoding['label'] = label2idx[example['label']]
    encoding['bbox'] = token_boxes
    encoding["gt_label"] = example['label']
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



def metric_calculation(u_true, u_pred):
    # label_names = ["OTHERS", "CS", "COO", "BOL", "AIR_WAY", "IC", "PL"]
    label_names = ['CI', 'BOL', 'COO', 'OTHERS', 'CS', 'IC', 'BOE', 'PL', 'AWB']
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
    true_labels = [idx2label[label_idx] for label_idx in true_labels]
    predicted_labels = [idx2label[label_idx] for label_idx in predicted_labels]

    confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=label_names)

    print("Confusion Matrix:")
    print(confusion_mat)
    return accuracy, precision, recall, f1


def convert_values(input_list, value_mapping):
    converted_list = [f'"{value_mapping.get(value, value)}' for value in input_list]
    return converted_list


def update_dataframe(df,imagename, value_list, label):
    # Check if the imagename already exists in the DataFrame
    if imagename in df['imagename'].values:
        # Append the new values to the existing list
        df.loc[df['imagename'] == imagename, 'values'].iloc[0].extend([value_list])
    else:
        # Add a new row with the imagename and the list of values
        df = pd.concat([df, pd.DataFrame({'imagename': [imagename], 'values': [[value_list]], "gt_label":[label]})], ignore_index=True)
    return df



if __name__=="__main__":
    dataset_path = "/home/ng6309/datascience/hridesh/layoutlmv1/classification/EVL_Bills/BILLS"
    ocr_path = '/home/ng6309/datascience/hridesh/OCR_Dec12'
    

    labels = [label for label in os.listdir(dataset_path)]

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
    device = "cuda"

    dataset = Dataset.from_dict(data)
    updated_dataset = dataset.map(apply_ocr)
    columns_names = ['image_path', 'label', 'words', 'bbox']  # Replace with your desired column names
    new_updated_dataset = pd.DataFrame(columns=columns_names)
    new_updated_dataset = filter_updated_data(updated_dataset, new_updated_dataset)
    # new_updated_dataset.to_csv("test3.csv",index=False)
    # new_updated_dataset = pd.read_csv("/home/ng6309/datascience/hridesh/layoutlmv1/test3.csv")
    new_updated_dataset = Dataset.from_pandas(new_updated_dataset)
    # encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example), features=training_features, batched=False, batch_size=1)
    encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example))
    encoded_dataset.set_format(type='torch', device=device)


    model = LayoutLMForSequenceClassification.from_pretrained("/home/ng6309/datascience/hridesh/layoutlmv1/classification/best_model_pre")
    model.to(device)
    model.eval()
    valid_dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=1, shuffle=True)

    df_aggregated = pd.DataFrame(columns=['imagename', 'values'])



    correctpred = 0
    wrongpred = 0
    imglabel = []
    imgname = []
    imgpred = []
    img_pth = []

    predicted_labels = []
    true_labels = []
    # For validation data
    validation_loss = 0.0
    validation_correct = 0
    for batch in tqdm(valid_dataloader):
        image_path = str(batch["image_path"][0])
        gt_label  = batch["gt_label"]
        labels = torch.tensor(batch["label"])
        t = labels.tolist()
        true_labels.append(t)
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # forward pass
        preds = torch.softmax(outputs.logits, dim=1).tolist()[0]
        df_aggregated = update_dataframe(df_aggregated,image_path, preds,gt_label[0])

        predictions = outputs.logits.argmax(-1)
        if predictions == labels:
            validation_correct = validation_correct+1
        x = predictions.tolist()
        predicted_labels.append(x)
        img_pth.append(image_path)

    with open("all_chunk_final_result.txt", 'w') as f:
        print("Validation Loss:", validation_loss / len(encoded_dataset))
        f.write(f"Validation Loss: :{validation_loss / len(encoded_dataset)}\n")

        validation_accuracy = 100 * validation_correct / len(encoded_dataset)
        print("Validation accuracy:", validation_accuracy)
        f.write(f"Validation accuracy: {validation_accuracy}\n")
        validation_loss = validation_loss / len(encoded_dataset)

    unlisted_true = [item[0] for item in true_labels]
    print(unlisted_true)
    unlisted_predicted = [item[0] for item in predicted_labels]
    print(unlisted_predicted)
    accuracy, precision, recall, f1 = metric_calculation(unlisted_true, unlisted_predicted)

    ground_truth = convert_values(unlisted_true, idx2label)
    pred_val = convert_values(unlisted_predicted, idx2label)

    with open("all_chunk_final_result.txt", 'a') as f:
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"f1: {f1}\n")
        f.write(classification_report([ground_truth], [pred_val]))

    result_dict = {
        'image_path': img_pth,
        'ground_truth': ground_truth,
        'prediction': pred_val
    }
    df = pd.DataFrame(result_dict)
    df['results'] = None
    for i in range(len(df)):
        if df['ground_truth'][i] == df['prediction'][i]:
            df['results'][i] = 1
        else:
            df['results'][i] = 0

    df.to_csv("all_chunk_eval_results.csv",index=False)



    df_aggregated["probs"]=df_aggregated["values"].apply(lambda x:np.mean(np.array(x), axis=0))
    df_aggregated["predicted_label"] = df_aggregated["probs"].apply(lambda x:idx2label[np.argmax(x)])

    predicted_labels = []
    true_labels = []

    correctpred = 0
    wrongpred = 0

    validation_correct = 0

    new_col_pred = []
    for i in range(len(df_aggregated)):
        values = df_aggregated.iloc[i]
        gt_label = values["gt_label"]
        predicted_label = values["predicted_label"]
        predicted_labels.append(predicted_label)
        true_labels.append(gt_label)
        if gt_label == predicted_label:
            validation_correct = validation_correct+1
            new_col_pred.append(1)
        else:
            new_col_pred.append(0)


    with open("aggregated_final_result.txt", 'w') as f:
        validation_accuracy = 100 * validation_correct / len(df_aggregated)
        print("Validation accuracy:", validation_accuracy)
        f.write(f"Validation accuracy: {validation_accuracy}\n")

    unlisted_true = [label2idx[item] for item in true_labels]
    unlisted_predicted = [label2idx[item] for item in predicted_labels]
    accuracy, precision, recall, f1 = metric_calculation(unlisted_true, unlisted_predicted)

    with open("aggregated_final_result.txt", 'a') as f:
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"f1: {f1}\n")
    
    df_aggregated["pred_1_0"] = new_col_pred

    df_aggregated.to_csv("aggregated_results.csv",index=False)
    print(" CODE EXECUTED SUCCESSFULLY___________________________!!!!!!!!!!!!!!!!!!!!")
