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
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
import os
from collections import Counter
from torch.utils.data import DataLoader
from os import listdir
from torch.utils.data import Dataset
from PIL import Image
from transformers import AdamW, LayoutLMForTokenClassification,LayoutLMTokenizer, LayoutLMConfig
import torch
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import warnings
import logging
from typing import List
from configparser import ConfigParser
from datetime import datetime
warnings.filterwarnings("ignore")
from seqeval.metrics import (
	classification_report,
	f1_score,
	precision_score,
	recall_score,
accuracy_score)
from torch.optim import SGD, RMSprop

#################################################################
# from config.prod_mapping import product_code_map, document_code_map
from datasets import load_dataset, Dataset, Features, Sequence, ClassLabel, Value, Array2D

global training_features
global tokenizer


tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")



def results_test(preds, out_label_ids, labels):
	preds = np.argmax(preds, axis=2)

	label_map = dict(enumerate(labels))

	out_label_list = [[] for _ in range(out_label_ids.shape[0])]
	preds_list = [[] for _ in range(out_label_ids.shape[0])]

	for i in range(out_label_ids.shape[0]):
		for j in range(out_label_ids.shape[1]):
			if out_label_ids[i, j] != -100:
				out_label_list[i].append(label_map[out_label_ids[i][j]])
				preds_list[i].append(label_map[preds[i][j]])

	results = {
		"precision": precision_score(out_label_list, preds_list),
		"recall": recall_score(out_label_list, preds_list),
		"f1": f1_score(out_label_list, preds_list)
	}
	return results, classification_report(out_label_list, preds_list)


def results_train(preds, out_label_ids, labels):
	preds = np.argmax(preds, axis=2)

	label_map = {i: label for i, label in enumerate(labels)}

	out_label_list = [[] for _ in range(out_label_ids.shape[0])]
	preds_list = [[] for _ in range(out_label_ids.shape[0])]

	for i in range(out_label_ids.shape[0]):
		for j in range(out_label_ids.shape[1]):
			if out_label_ids[i, j] != -100:
				out_label_list[i].append(label_map[out_label_ids[i][j]])
				preds_list[i].append(label_map[preds[i][j]])

	results = {
		"precision": precision_score(out_label_list, preds_list),
		"recall": recall_score(out_label_list, preds_list),
		"f1": f1_score(out_label_list, preds_list),
	}
	return results, classification_report(out_label_list, preds_list)


def set_basic_config_for_logging(folder_path, filename: str = None):
	"""    
	Set the basic config for logging python program.   
	:return: None   
	"""    
	# Create and configure logger    
	log_file_path = os.path.join(folder_path, f"{filename}.log")
	logging.basicConfig(filename=log_file_path, format='%(asctime)s %(message)s',
						filemode='w')
	
def get_logger_object_and_setting_the_loglevel():
	"""    get the logger object and set the loglevel for the logger object    
	:return: Logger Object    
	"""    
	# Creating an object    
	logger_object = logging.getLogger()
	# Setting the threshold of logger to DEBUG    
	logger_object.setLevel(logging.DEBUG)
	return logger_object


def encode_example(example, max_seq_length=512):
	token_boxes = []
	aligned_labels = []
	for word, box, label in zip(example['words'], example['bbox'], example['label']):
		word_tokens = tokenizer.tokenize(word)
		token_boxes.extend([box] * len(word_tokens))
		aligned_labels.append(label2id[label])
		aligned_labels.extend([-100 for _ in range(len(word_tokens)-1)])

	special_tokens_count = 2 
	if len(token_boxes) > max_seq_length - special_tokens_count:
		token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
		aligned_labels = aligned_labels[:(max_seq_length - special_tokens_count)]
		
	aligned_labels = [0] + aligned_labels + [0]
	token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

	encoding = tokenizer(" ".join(example['words']), padding='max_length', truncation=True)
	
	pad_token_box = [0, 0, 0, 0]
	padding_length = max_seq_length - len(tokenizer(' '.join(example['words']), truncation=True)["input_ids"])
	token_boxes += [pad_token_box] * padding_length
	aligned_labels += [0] * padding_length
 
	encoding['bboxes'] = token_boxes
	encoding['ner_tags'] = aligned_labels
	return encoding


def filter_updated_data(updated_dataset, new_updated_dataset):
	for i in range(len(updated_dataset)):
		if len(updated_dataset[i]['words']) !=0 and len(updated_dataset[i]['bbox']) !=0:
			new_row = {'image_path': updated_dataset[i]['image_path'], 'label': updated_dataset[i]['label'], 'words': updated_dataset[i]['words'], 'bbox': updated_dataset[i]['bbox']}
			new_row_df = pd.DataFrame([new_row])
			new_updated_dataset = pd.concat([new_updated_dataset, new_row_df], ignore_index=True)
		return new_updated_dataset

def train_test_split_base(dataset, flag):
	if flag=='train':
		# updated_dataset = Dataset.from_dict(dataset)
		# columns_names = ['image_path', 'label', 'words', 'bbox']  # Replace with your desired column names
		# new_updated_dataset = pd.DataFrame(columns=columns_names)
		# new_updated_dataset = filter_updated_data(updated_dataset, new_updated_dataset)
		# new_updated_dataset.to_csv("test3.csv",index=False)
		# new_updated_dataset = pd.read_csv("/home/ng6309/datascience/hridesh/layoutlmv1/test3.csv")
		new_updated_dataset = Dataset.from_pandas(dataset)
		# encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example), features=training_features, batched=False, batch_size=1)
		encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example))
		print(encoded_dataset)
		encoded_dataset.set_format(type='torch', columns=['input_ids', 'bboxes', 'attention_mask', 'token_type_ids', 'ner_tags'])
		return encoded_dataset
	elif flag=='test':
		# updated_dataset = Dataset.from_dict(dataset)
		# columns_names = ['image_path', 'label', 'words', 'bbox']  # Replace with your desired column names
		# new_updated_dataset = pd.DataFrame(columns=columns_names)
		# new_updated_dataset = filter_updated_data(updated_dataset, new_updated_dataset)
		new_updated_dataset = Dataset.from_pandas(dataset)
		# encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example), features=training_features, batched=False, batch_size=1)
		encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example))
		encoded_dataset.set_format(type='torch', columns=['input_ids', 'bboxes', 'attention_mask', 'token_type_ids', 'ner_tags'])
		return encoded_dataset    


global input_json_dict

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


def modify_llm_config(o_config,label2idx):
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


log_dir = "logs"  # Directory to store the TensorBoard logs


product_code_map : dict = {
"lc" : "LetterOfCredit",
"bg" : "BankGuarantees",
"tp" : "TransportDocument"
}


document_code_map : dict = {
"po": "PurchaseOrder",
"pi": "PerformaInvoice",
"bgc": "BGCancellation",
"lca": "LCApplication",
"bol" : "BillOfLadding",
"awb" : "AirWayBill",
"coo" : "CertificateOfOrigin",
"pl" : "PackingList",
"ic" : "InsuranceCertificate",
"ci" : "CommercialInvoice",
"boe": "BillOfExchange",
"cs" : "CoveringSchedule"
}

# product config

# data folder path

prod_code = "tp"
doc_code_ = "awb"

train_writer = SummaryWriter(log_dir=f'''logs/{doc_code_}/{doc_code_}/{"_".join(str(datetime.now()).split(" "))}/train''')
test_writer = SummaryWriter(log_dir=f'''logs/{doc_code_}/{doc_code_}/{"_".join(str(datetime.now()).split(" "))}/test''')
best_writer = SummaryWriter(log_dir=f'''logs/{doc_code_}/{doc_code_}/{"_".join(str(datetime.now()).split(" "))}/best''')

doc_code = document_code_map[doc_code_]
folder_path = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/AWB"
# folder_path = '/New_Volume/Rakesh/DATA_LMV2/LMV2_BASE/AWB'
print("==================Trade Finance Solutions===================")
# print("Product Code: {product_code}")
# print("Documenry Code: {doc_code}")
print(f"folder_path: {folder_path}")
set_basic_config_for_logging(folder_path, filename="train_words_count")
logger = get_logger_object_and_setting_the_loglevel()
train = pd.read_pickle(os.path.join(folder_path, 'train.pkl'))
test = pd.read_pickle(os.path.join(folder_path, 'test.pkl'))
all_labels = [item for sublist in train[2] for item in sublist] + [item for sublist in test[2] for item in sublist]
Counter(all_labels)
label_new = dict(Counter(all_labels))
labels = list(set(all_labels))
print(labels)
print(len(labels))

with open(os.path.join(folder_path, "classes.txt"), "w") as f:
	f.write(str(labels))
f.close()

#same count in labels and classes (+1 for others)
global label2id
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
print(label2id,flush=True)
print(id2label,flush=True)
train_df = pd.DataFrame({
	"image_path":train[0],
	"words":train[1],
	"label":train[2],
	"bbox":train[3],
})

test_df = pd.DataFrame({
	"image_path":test[0],
	"words":test[1],
	"label":test[2],
	"bbox":test[3],
})

train_data = train_test_split_base(train_df , 'train')
valid_data = train_test_split_base(test_df , 'test')

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=True)
original_config = LayoutLMConfig.from_pretrained("microsoft/layoutlm-base-uncased")
# Modify the configuration as needed
modified_config = modify_llm_config(original_config,id2label)

model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=len(id2label))
device = "cuda"
model.to(device)


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

print(device)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
labels = list(set(all_labels))
global_step = 0
num_train_epochs = 40
preds_val = None
out_label_ids = None
best_loss=None
best_precision=None
best_recall=None
best_f1=None
steps = []
losses = []
training_loss = {}  
validation_loss = {}
# put the model in training mode
model.train()
best_model_flag_high = False
best_model_flag_low = False
for epoch in range(num_train_epochs):
	print("Epoch:", epoch)
	for batch in tqdm(train_dataloader):
		labels = batch["ner_tags"].to(device)
		# input_ids = batch['input_ids'].to(device)
		# bbox = batch['bbox'].to(device)
		# image = batch['image'].to(device)
		# attention_mask = batch['attention_mask'].to(device)
		# token_type_ids = batch['token_type_ids'].to(device)
		# labels = batch['labels'].to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(
			input_ids=batch["input_ids"].to(device), bbox=batch["bboxes"].to(device), 
			attention_mask=batch["attention_mask"].to(device), 
			token_type_ids=batch["token_type_ids"].to(device), 
			labels=labels
		)

		loss = outputs.loss

		# print loss every epoch
		if (global_step + 1) % len(train_dataloader) == 0 or global_step == 0:
			print(f"Loss after {global_step} steps: {loss.item()}")
			steps.append(global_step)
			losses.append(float(loss.item()))
		loss.backward()
		optimizer.step()
		global_step += 1
	# model.eval()
	training_loss[epoch] = loss 
	val_loss = 0.0
	preds_val = None
	for batch in tqdm(valid_dataloader, desc="Evaluating"):
		with torch.no_grad():
			labels = batch["ner_tags"].to(device)
			outputs = model(
			input_ids=batch["input_ids"].to(device), bbox=batch["bboxes"].to(device), 
			attention_mask=batch["attention_mask"].to(device), 
			token_type_ids=batch["token_type_ids"].to(device), 
			labels=labels
		)

			# input_ids = batch['input_ids'].to(device)
			# bbox = batch['bbox'].to(device)
			# image = batch['image'].to(device)
			# attention_mask = batch['attention_mask'].to(device)
			# token_type_ids = batch['token_type_ids'].to(device)
			# labels = batch['labels'].to(device)

			# outputs = model(input_ids=input_ids,
			# 				bbox=bbox,
			# 				image=image,
			# 				attention_mask=attention_mask,
			# 				token_type_ids=token_type_ids,
			# 				labels=labels)
			val_los= outputs.loss
			# print(f'batch validataion loss: {val_los}')
			val_loss += val_los.item()

			if preds_val is None:
				preds_val = outputs.logits.detach().cpu().numpy()
				out_label_ids = batch["ner_tags"].detach().cpu().numpy()
			else:
				preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
				out_label_ids = np.append(
					out_label_ids, batch["ner_tags"].detach().cpu().numpy(), axis=0)

	labels = list(set(all_labels))
	val_result, class_report = results_test(preds_val, out_label_ids, labels)

	
	print(f"precison: {val_result['precision']}")
	print(f"recall: {val_result['recall']}")
	print(f"f1: {val_result['f1']}")
	print('+++++++++++++++++++++++++++++++++++++++++++')
	val_loss= val_loss /len(valid_dataloader)
	validation_loss[epoch] = val_loss
	print(f'final validation loss:{val_loss}')
	# print(val_result)
	# with train_writer.as_default():
	#     tf.summary.scalar("train loss ", loss.detach().cpu(), step=epoch)
	# with test_writer.as_default():
	#     tf.summary.scalar("Validation Loss", val_loss, step=epoch)   
	#precision, recall values need to log
	
	train_writer.add_scalar("train loss", loss.detach(), epoch)
	test_writer.add_scalar("val loss", val_los, epoch)
	
	
	precision = val_result['precision']
	recall = val_result['recall']
	f1= val_result['f1']
	
	# val metrics
	test_writer.add_scalar("val precision ", precision, epoch)
	test_writer.add_scalar("val f1",f1, epoch)
	test_writer.add_scalar("val recall ", recall, epoch)
	
	if epoch % 5==0:
		path_intermediate = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/AWB/intermediate_path/"
		model.save_pretrained(path_intermediate+"model_{}".format(epoch))
	
	if  best_loss is None:
		best_loss=val_loss
	if best_precision is None:
		best_precision = precision
		best_recall = recall
		best_f1 = f1
	# print(f"best precison: {best_precision}")
	# print(f"best recall: {best_recall}")
	
	if val_loss < best_loss and f1 > best_f1 and recall > best_recall:
		best_model_flag_high = True
		best_loss = val_loss
		best_precision = precision
		best_recall = recall
		best_f1 = f1
		name = "Best_Model"

		if not os.path.exists(os.path.join(folder_path, name)):
			os.mkdir(os.path.join(folder_path, name))
		
		print(f'Model is {epoch} saving +++++++++++++++++++++++++++++++++')
		with open(os.path.join(folder_path, "model_saving_info.txt"), 'a') as f:
			f.write(f"Model is {epoch} saving +++++++++++++++++++++++++++++++++\n")

		# with best_train_test_writer.as_default():
		#     tf.summary.scalar("Best train loss ", best_loss, step=epoch)
		# with best_train_test_writer.as_default():
		#     tf.summary.scalar("best_precision ", best_precision, step=epoch) 
		# with best_train_test_writer.as_default():
		#     tf.summary.scalar("best_f1",best_f1, step=epoch) 
		# with best_train_test_writer.as_default():
		#     tf.summary.scalar("best_recall ", best_recall, step=epoch) 
		
		# best metrics 
		best_writer.add_scalar("Best loss ", best_loss, epoch)
		best_writer.add_scalar("best precision ", best_precision, epoch)
		best_writer.add_scalar("best f1",best_f1, epoch)
		best_writer.add_scalar("best recall ", best_recall, epoch)
		
		print(f"best Validation Loss: {best_loss}" )
		print("best Precision:", best_precision)
		print("best Recall:", best_recall) 
		print("best f1:", best_f1)
		model.save_pretrained(os.path.join(folder_path, name))

	if val_loss < best_loss:
		best_model_flag_low = True
		best_loss = val_loss
		best_writer.add_scalar("Best loss ", best_loss, epoch)
		best_writer.add_scalar("best precision ", best_precision, epoch)
		best_writer.add_scalar("best f1",best_f1, epoch)
		best_writer.add_scalar("best recall ", best_recall, epoch)
		name = "Best_Model_low"
		best_model_low = model
		
if not best_model_flag_high and best_model_flag_low:
	best_model_low.save_pretrained(os.path.join(folder_path, name))
	


#give best model path here
if best_model_flag_high:
	model_path = f"{folder_path}/Best_Model"
elif best_model_flag_low:
	model_path =  f"{folder_path}/Best_Model_low"
else:
	exit("no best model exist")

try:
	model = LayoutLMForTokenClassification.from_pretrained(
			pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
			config=os.path.join(model_path, 'config.json'))
except:
	model = LayoutLMForTokenClassification.from_pretrained(model_path)

print(training_loss)
with open(os.path.join(folder_path, "training_loss.txt"), 'w') as f:
	for key, value in training_loss.items():
		f.write(f"{key}: {value}\n") 

print(validation_loss)
with open(os.path.join(folder_path, "validation_loss.txt"), 'w') as f:
	for key, value in validation_loss.items():
		f.write(f"{key}: {value}\n")        
		
print('woo!,Model Training has done successfully')