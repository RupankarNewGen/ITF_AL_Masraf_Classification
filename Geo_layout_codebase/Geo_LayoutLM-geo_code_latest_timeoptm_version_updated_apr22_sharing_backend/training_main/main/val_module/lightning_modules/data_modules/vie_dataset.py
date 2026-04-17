# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils import get_class_names


class VIEDataset(Dataset):
    def __init__(
        self,
        dataset,
        task,
        backbone_type,
        model_head,
        dataset_root_path,
        tokenizer,
        max_seq_length=512,
        max_block_num=256,
        img_h=768,
        img_w=768,
        mode=None,
    ):
        self.dataset = dataset
        self.task = task
        self.backbone_type = backbone_type
        self.model_head = model_head

        self.dataset_root_path = dataset_root_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_block_num = max_block_num
        self.img_h = img_h
        self.img_w = img_w
        self.mode = mode
        print(f"data_root: {dataset_root_path}")

        if getattr(self.tokenizer, "vocab", None) is not None:
            self.pad_token_id = self.tokenizer.vocab["[PAD]"]
            self.cls_token_id = self.tokenizer.vocab["[CLS]"]
            self.sep_token_id = self.tokenizer.vocab["[SEP]"]
            self.unk_token_id = self.tokenizer.vocab["[UNK]"]
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.cls_token_id = self.tokenizer.cls_token_id
            self.sep_token_id = self.tokenizer.sep_token_id
            self.unk_token_id = self.tokenizer.unk_token_id

        self.examples = self._load_examples()
        # print(f'examples: {len(self.examples)}')
        # exit('++++++++++++=')

        self.class_names = get_class_names(self.dataset_root_path)
        # print(f'class names:{self.class_names}')
        self.class_idx_dic = dict(
            [(class_name, idx) for idx, class_name in enumerate(self.class_names)]
        )
        # print(f'classes in dict: {self.class_idx_dic}')
        # exit('+++++++++++++++++')
        self.bio_class_names = ["O"]
        for class_name in self.class_names:
            # if not class_name.startswith('O'):
            if class_name.upper() != 'O':
                self.bio_class_names.extend([f"B-{class_name}", f"I-{class_name}"])
        self.bio_class_idx_dic = dict(
            [
                (bio_class_name, idx)
                for idx, bio_class_name in enumerate(self.bio_class_names)
            ]
        )
        print(f'classes in dict with i and b classes: {self.bio_class_idx_dic}')
        # exit('+++++++++++++++')

    def _load_examples(self):
        examples = []
        with open(
            os.path.join(self.dataset_root_path, f"preprocessed_files_{self.mode}.txt"),        
            "r",
            encoding="utf-8",
        ) as fp:
            for line in fp.readlines():
                preprocessed_file = os.path.join(self.dataset_root_path, line.strip())
                examples.append(
                    json.load(open(preprocessed_file, "r", encoding="utf-8"))
                )

        return examples

    def __len__(self):
        return len(self.examples)

    def _getitem_geo(self, idx):
        json_obj = self.examples[idx]
        
        return_dict = {}

        width = json_obj["meta"]["imageSize"]["width"]
        height = json_obj["meta"]["imageSize"]["height"]

        img_path = os.path.join(self.dataset_root_path, json_obj["meta"]["image_path"])

        image = cv2.resize(cv2.imread(img_path, 1), (self.img_w, self.img_h))
        image = image.astype("float32").transpose(2, 0, 1)

        return_dict["image_path"] = img_path
        return_dict["image"] = image
        return_dict["size_raw"] = np.array([width, height])

        return_dict["input_ids"] = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        return_dict["bbox_4p_normalized"] = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        return_dict["attention_mask"] = np.zeros(self.max_seq_length, dtype=int)
        return_dict["first_token_idxes"] = np.zeros(self.max_block_num, dtype=int)
        return_dict["block_mask"] = np.zeros(self.max_block_num, dtype=int)
        return_dict["bbox"] = np.zeros((self.max_seq_length, 4), dtype=np.float32)
        return_dict["line_rank_id"] = np.zeros(self.max_seq_length, dtype="int32")
        return_dict["line_rank_inner_id"] = np.ones(self.max_seq_length, dtype="int32")

        return_dict["are_box_first_tokens"] = np.zeros(self.max_seq_length, dtype=np.bool_)
        return_dict["bio_labels"] = np.zeros(self.max_seq_length, dtype=int)
        return_dict["el_labels_seq"] = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
        return_dict["el_label_seq_mask"] = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
        return_dict["el_labels_blk"] = np.zeros((self.max_block_num, self.max_block_num), dtype=np.float32)
        return_dict["el_label_blk_mask"] = np.zeros((self.max_block_num, self.max_block_num), dtype=np.float32)

        list_tokens = []
        list_bbs = [] # word boxes
        list_blk_bbs = [] # block boxes
        box2token_span_map = []

        box_to_token_indices = []
        cum_token_idx = 0

        cls_bbs = [0.0] * 8
        cls_bbs_blk = [0] * 4

        for word_idx, word in enumerate(json_obj["words"]):
            this_box_token_indices = []

            tokens = word["tokens"]
            bb = word["boundingBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)

            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                print(f'the data has truncated+++++++++++++++++++++')
                # print(f"text: {word['text']}")
                # print(f'tokens: {tokens}')
                curr_lst_tokens= list_tokens
                
                break # truncation for long documents

            box2token_span_map.append(
                [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
            )  # including st_idx, start from 1
            list_tokens += tokens
            # min, max clipping
            for coord_idx in range(4):
                bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
                bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

            bb = list(itertools.chain(*bb))
            bbs = [bb for _ in range(len(tokens))]

            for _ in tokens:
                cum_token_idx += 1
                this_box_token_indices.append(cum_token_idx) # start from 1

            list_bbs.extend(bbs)
            box_to_token_indices.append(this_box_token_indices)
        # exit('++++++++++++++++s')
        # print(f'list tokens: {list_tokens}')
        # print(len(list_tokens))
        sep_bbs = [width, height] * 4
        sep_bbs_blk = [width, height] * 2

        first_token_idx_list = json_obj['blocks']['first_token_idx_list'][:self.max_block_num]
        # print(first_token_idx_list[-1])
        # exit('+++++++++++++++++++')
        if first_token_idx_list[-1] > len(list_tokens):
            blk_length = self.max_block_num
            for blk_id, first_token_idx in enumerate(first_token_idx_list):
                if first_token_idx > len(list_tokens):
                    blk_length = blk_id
                    break
            first_token_idx_list = first_token_idx_list[:blk_length]
            
        first_token_ext = first_token_idx_list + [len(list_tokens) + 1]
        # print(first_token_ext)
        # exit('+++++++++++++++')
        line_id = 1
        for blk_idx in range(len(first_token_ext) - 1):
            token_span = first_token_ext[blk_idx+1] - first_token_ext[blk_idx]
            # block box
            bb_blk = json_obj['blocks']['boxes'][blk_idx]
            bb_blk[0] = max(0, min(bb_blk[0], width))
            bb_blk[1] = max(0, min(bb_blk[1], height))
            bb_blk[2] = max(0, min(bb_blk[2], width))
            bb_blk[3] = max(0, min(bb_blk[3], height))
            list_blk_bbs.extend([bb_blk for _ in range(token_span)])
            # line_rank_id
            return_dict["line_rank_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = line_id
            line_id += 1
            # line_rank_inner_id
            if token_span > 1:
                return_dict["line_rank_inner_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = [1] + [2] * (token_span - 2) + [3]

        # For [CLS] and [SEP]
        list_tokens = (
            [self.cls_token_id]
            + list_tokens[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(list_bbs) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            list_bbs = [cls_bbs] + [sep_bbs]
            list_blk_bbs = [cls_bbs_blk] + [sep_bbs_blk]
        else:  # len(list_bbs) > 0
            list_bbs = [cls_bbs] + list_bbs[: self.max_seq_length - 2] + [sep_bbs]
            list_blk_bbs = [cls_bbs_blk] + list_blk_bbs[: self.max_seq_length - 2] + [sep_bbs_blk]

        len_list_tokens = len(list_tokens)
        len_blocks = len(first_token_idx_list)
        # print(len_list_tokens)
        # exit('++++++++++++++++++)))')
        # print(len_blocks)
        # exit('++++++++++++=')
        return_dict["input_ids"][:len_list_tokens] = list_tokens
        return_dict["attention_mask"][:len_list_tokens] = 1
        return_dict["first_token_idxes"][:len(first_token_idx_list)] = first_token_idx_list
        return_dict["block_mask"][:len_blocks] = 1
        return_dict["line_rank_inner_id"] = return_dict["line_rank_inner_id"] * return_dict["attention_mask"]

        bbox_4p_normalized = return_dict["bbox_4p_normalized"]
        bbox_4p_normalized[:len_list_tokens, :] = list_bbs

        # bounding box normalization -> [0, 1]
        bbox_4p_normalized[:, [0, 2, 4, 6]] = bbox_4p_normalized[:, [0, 2, 4, 6]] / width
        bbox_4p_normalized[:, [1, 3, 5, 7]] = bbox_4p_normalized[:, [1, 3, 5, 7]] / height

        if self.backbone_type == "layoutlm":
            bbox_4p_normalized = bbox_4p_normalized[:, [0, 1, 4, 5]]
            bbox_4p_normalized = bbox_4p_normalized * 1000
            bbox_4p_normalized = bbox_4p_normalized.astype(int)

        return_dict["bbox_4p_normalized"] = bbox_4p_normalized
        bbox = return_dict["bbox"]

        bbox[:len_list_tokens, :] = list_blk_bbs
        # bbox -> [0, 1000)
        bbox[:, [0, 2]] = bbox[:, [0, 2]] / width * 1000
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / height * 1000
        bbox = bbox.astype(int)
        return_dict["bbox"] = bbox

        st_indices = [
            indices[0]
            for indices in box_to_token_indices
            if indices[0] < self.max_seq_length
        ]
        return_dict["are_box_first_tokens"][st_indices] = True

        # Label for tagging
        # classes_dic = json_obj["parse"]["class"]
        # for class_name in self.class_names:
        #     # if class_name == "O":
        #     #     continue
        #     if class_name not in classes_dic:
        #         continue

        #     for word_list in classes_dic[class_name]:
        #         # At first, connect the class and the first box
        #         is_first, last_word_idx = True, -1
        #         for word_idx in word_list:
        #             if word_idx >= len(box2token_span_map):
        #                 break
        #             box2token_span_start, box2token_span_end = box2token_span_map[
        #                 word_idx
        #             ]
        #             for converted_word_idx in range(
        #                 box2token_span_start, box2token_span_end
        #             ):
        #                 if converted_word_idx >= self.max_seq_length:
        #                     break

        #                 if class_name == 'O':
        #                     return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
        #                         "O"
        #                     ]
        #                     continue
        #                 if is_first:
        #                     return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
        #                         f"B-{class_name}"
        #                     ]
        #                     is_first = False
        #                 else:
        #                     return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
        #                         f"I-{class_name}"
        #                     ]
        return_dict["bio_labels"][0] = -100
        return_dict["bio_labels"][len_list_tokens:] = -100
        # Label for linking
        # relations = json_obj["parse"]["relations"]

        # for relation in relations:
        #     if relation[0] >= len(box2token_span_map) or relation[1] >= len(box2token_span_map):
        #         continue
        #     if (
        #         box2token_span_map[relation[0]][0] >= self.max_seq_length
        #         or box2token_span_map[relation[1]][0] >= self.max_seq_length
        #     ):
        #         continue

        #     word_from = box2token_span_map[relation[0]][0]
        #     word_to = box2token_span_map[relation[1]][0]
        #     return_dict["el_labels_seq"][word_to][word_from] = 1.0
        # return_dict["el_label_seq_mask"][1:len_list_tokens, 1:len_list_tokens] = 1.0
        # B_idx_all = np.array(first_token_idx_list)
        # return_dict["el_labels_blk"][:len(first_token_idx_list), :len(first_token_idx_list)] = \
        #     return_dict["el_labels_seq"][B_idx_all[:, np.newaxis], B_idx_all[np.newaxis, :]]
        # return_dict["el_label_blk_mask"][:len_blocks, :len_blocks] = 1.0

        for k in return_dict.keys():
            if isinstance(return_dict[k], np.ndarray):
                return_dict[k] = torch.from_numpy(return_dict[k])

        return return_dict

    def __getitem__(self, idx):
        if self.model_head == "vie":
            # print(f'idx value:{idx}')
            return_dict = self._getitem_geo(idx)
            # print(return_dict)
            # exit('++++++++++++++')

        else:
            raise ValueError(f"Unknown self.model_head={self.model_head}")

        return return_dict




class VIEDatasetInfer(Dataset):
    def __init__(
        self,
        dataset,
        task,
        backbone_type,
        model_head,
        dataset_root_path,
        tokenizer,
        max_seq_length=512,
        max_block_num=256,
        img_h=768,
        img_w=768,
        mode=None,
    ):
        self.dataset = dataset
        self.task = task
        self.backbone_type = backbone_type
        self.model_head = model_head

        self.dataset_root_path = dataset_root_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_block_num = max_block_num
        self.img_h = img_h
        self.img_w = img_w
        self.mode = mode
        print(f"data_root: {dataset_root_path}")

        if getattr(self.tokenizer, "vocab", None) is not None:
            self.pad_token_id = self.tokenizer.vocab["[PAD]"]
            self.cls_token_id = self.tokenizer.vocab["[CLS]"]
            self.sep_token_id = self.tokenizer.vocab["[SEP]"]
            self.unk_token_id = self.tokenizer.vocab["[UNK]"]
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.cls_token_id = self.tokenizer.cls_token_id
            self.sep_token_id = self.tokenizer.sep_token_id
            self.unk_token_id = self.tokenizer.unk_token_id

        self.examples = self._load_examples()
        print(f'examples: {len(self.examples)}')
        # exit('++++++++++++=')

        self.class_names = get_class_names(self.dataset_root_path)
        # print(f'class names:{self.class_names}')
        self.class_idx_dic = dict(
            [(class_name, idx) for idx, class_name in enumerate(self.class_names)]
        )
        # print(f'classes in dict: {self.class_idx_dic}')
        # exit('+++++++++++++++++')
        self.bio_class_names = ["O"]
        for class_name in self.class_names:
            # if not class_name.startswith('O'):
            if class_name.upper() != 'O':
                self.bio_class_names.extend([f"B-{class_name}", f"I-{class_name}"])
        self.bio_class_idx_dic = dict(
            [
                (bio_class_name, idx)
                for idx, bio_class_name in enumerate(self.bio_class_names)
            ]
        )
        print(f'classes in dict with i and b classes: {self.bio_class_idx_dic}')
        # exit('+++++++++++++++')

    def _load_examples(self):
        examples = []
        with open(
            os.path.join(self.dataset_root_path, f"preprocessed_files_{self.mode}.txt"),        
            "r",
            encoding="utf-8",
        ) as fp:
            for line in fp.readlines():
                preprocessed_file = os.path.join(self.dataset_root_path, line.strip())
                examples.append(
                    json.load(open(preprocessed_file, "r", encoding="utf-8"))
                )

        return examples

    def __handlingTruncData__(self,json_obj):
        max_sequence_len_thresh= 350
        lst_size=[]
        start_index= None
        end_index= None
        track_word_corrds=[]
        all_tokens=[]
        all_boxes= []
        divide_block=[]
        counter=0
        for word_idx, word in enumerate(json_obj["words"]):
            tokens = word["tokens"]
            bb = word["boundingBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)

            if len(lst_size) + len(tokens) > max_sequence_len_thresh:
                print(f'number of tokens in each chunk: {len(lst_size) + len(tokens)}')
                counter+=1
                # print('entered: ++++++++++++++++++')
                if start_index is None:
                    track_word_corrds.append({0:word_idx})
                    start_index= 0
                    end_index= word_idx
                    divide_block.append({-1:json_obj['blocks']['first_token_idx_list'].index(len(all_tokens)+len(lst_size))})
                    all_tokens+=lst_size
                    lst_size.clear()
                    lst_size += tokens
                else:
                    start_index= end_index
                    end_index= word_idx
                    track_word_corrds.append({start_index:end_index})
                    divide_block.append({json_obj['blocks']['first_token_idx_list'].index(len(all_tokens)):json_obj['blocks']['first_token_idx_list'].index(len(all_tokens)+len(lst_size))})
                    all_tokens+= lst_size
                    lst_size.clear()
                    lst_size += tokens
        
            lst_size += tokens
        # print(f'lst elements: {lst_size}')
        # print(f'lst elements: {len(lst_size)}')
        divide_block.append({json_obj['blocks']['first_token_idx_list'].index(len(all_tokens)):json_obj['blocks']['first_token_idx_list'].index(json_obj['blocks']['first_token_idx_list'][-2])})
        all_tokens+= lst_size
        if counter>1:
            track_word_corrds.append({end_index:len(json_obj["words"])})
        # block_size= []
        # track_block=[]
        # start_block_index=0
        # end_block_index=0
        # for token_index, token_ids in enumerate(json_obj['blocks']['first_token_idx_list']):
        #     # print(token_ids)
        #     # exit('++++++++++++++++==')
        #     block_size.append(token_ids)
        #     if len(block_size)== self.max_block_num:
        #         divide_block.append({0:token_index+1})
        #         track_block.append(token_index+1)
        #         block_size.clear()
        # if len(block_size)< self.max_block_num and len(track_block)>0:
        #     divide_block.append({track_block[-1]:len(json_obj['blocks']['first_token_idx_list'])})
        
        print(f'word coordinates: {track_word_corrds}')
        print(f'block divide element: {divide_block}')
        # exit('+++++++++++++=')
        return track_word_corrds, divide_block
      


    def __len__(self):
        return len(self.examples)
    
    def __preprocess__(self, json_obj, words:dict, blocks:dict, processed_tokens,count):
        return_dict = {}
        if len(json_obj['blocks']['first_token_idx_list'])== len(json_obj['blocks']['boxes']):
            print('both are equal +++++++++++++++++++++++')
            print(len(json_obj["words"]))
        else:
            print('not equal ++++++++++++++++++++++++++++')
        # exit('++++++++++++++++++++++')
        width = json_obj["meta"]["imageSize"]["width"]
        height = json_obj["meta"]["imageSize"]["height"]

        img_path = os.path.join(self.dataset_root_path, json_obj["meta"]["image_path"])

        image = cv2.resize(cv2.imread(img_path, 1), (self.img_w, self.img_h))
        image = image.astype("float32").transpose(2, 0, 1)

        return_dict["image_path"] = img_path
        return_dict["image"] = image
        return_dict["size_raw"] = np.array([width, height])

        return_dict["input_ids"] = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        return_dict["bbox_4p_normalized"] = np.zeros((self.max_seq_length, 8), dtype=np.float32)
        return_dict["attention_mask"] = np.zeros(self.max_seq_length, dtype=int)
        return_dict["first_token_idxes"] = np.zeros(self.max_block_num, dtype=int)
        return_dict["block_mask"] = np.zeros(self.max_block_num, dtype=int)
        return_dict["bbox"] = np.zeros((self.max_seq_length, 4), dtype=np.float32)
        return_dict["line_rank_id"] = np.zeros(self.max_seq_length, dtype="int32")
        return_dict["line_rank_inner_id"] = np.ones(self.max_seq_length, dtype="int32")

        return_dict["are_box_first_tokens"] = np.zeros(self.max_seq_length, dtype=np.bool_)
        return_dict["bio_labels"] = np.zeros(self.max_seq_length, dtype=int)
        return_dict["el_labels_seq"] = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
        return_dict["el_label_seq_mask"] = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
        return_dict["el_labels_blk"] = np.zeros((self.max_block_num, self.max_block_num), dtype=np.float32)
        return_dict["el_label_blk_mask"] = np.zeros((self.max_block_num, self.max_block_num), dtype=np.float32)

        list_tokens = []
        list_bbs = [] # word boxes
        list_blk_bbs = [] # block boxes
        box2token_span_map = []

        box_to_token_indices = []
        cum_token_idx = 0

        cls_bbs = [0.0] * 8
        cls_bbs_blk = [0] * 4
        # print(words)
        for key, value in words.items():
            start_index= key
            end_index= value
        for key, value in blocks.items():
            block_start_index= key
            block_end_index= value
        print(f'star indix: {start_index} and end indix: {end_index}')
        print(f'star indix: {block_start_index} and end indix: {block_end_index}')
        # exit('++++++++++++++++==')
        for word_idx, word in enumerate(json_obj["words"][start_index:end_index],start=start_index):
            # print(word_idx)
            # exit('+++++++++++++++++')
            this_box_token_indices = []

            tokens = word["tokens"]
            bb = word["boundingBox"]
            if len(tokens) == 0:
                tokens.append(self.unk_token_id)

            if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                print(f"text: {word['text']}")
                print(f'tokens: {tokens}')
                print(f'index: {word_idx}')
                # 
                
                
                break # truncation for long documents

            box2token_span_map.append(
                [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
            )  # including st_idx, start from 1
            list_tokens += tokens
            # min, max clipping
            for coord_idx in range(4):
                bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
                bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

            bb = list(itertools.chain(*bb))
            bbs = [bb for _ in range(len(tokens))]

            for _ in tokens:
                cum_token_idx += 1
                this_box_token_indices.append(cum_token_idx) # start from 1

            list_bbs.extend(bbs)
            box_to_token_indices.append(this_box_token_indices)
        print(f'list tokens: {list_tokens}')
        # print(f'list bounding boxes:{len(list_bbs)}')
        print(len(list_tokens))
        len_of_tokens= len(list_tokens)
        # exit('++++++++++++++++s')
        sep_bbs = [width, height] * 4
        sep_bbs_blk = [width, height] * 2

        first_token_idx_list = json_obj['blocks']['first_token_idx_list'][block_start_index+1:block_end_index+1]
        # print(f'last token box: {first_token_idx_list[-1]}')
        print(f'token id list: {first_token_idx_list}')
        print(f'len of the blocks: {len(first_token_idx_list)}')
        # print(f'processed_tokens:{processed_tokens}')
        # # exit('+++++++++++++++++++')
        # if first_token_idx_list[-1] > len(list_tokens)+processed_tokens:
        #     blk_length = self.max_block_num
        #     for blk_id, first_token_idx in enumerate(first_token_idx_list):
        #         if first_token_idx > len(list_tokens):
        #             blk_length = blk_id
        #             break
        #     first_token_idx_list = first_token_idx_list[:blk_length]
        if processed_tokens==0:
            first_token_ext = first_token_idx_list+[json_obj['blocks']['first_token_idx_list'][block_end_index+1]]
        else:
            print(f'first token idex: {first_token_idx_list}')
            # print(f'first_token_idx_list in else: {first_token_idx_list}')
            first_token_idx_list.append(json_obj['blocks']['first_token_idx_list'][block_end_index+1])
            first_token_ext= first_token_idx_list

        # first_token_ext = first_token_idx_list + [len(list_tokens)+processed_tokens + 1]
        print(f"extension token box: {first_token_ext}")
        # exit('+++++++++++++++')
        line_id = 1
        # index= json_obj['blocks']['first_token_idx_list'].index(processed_tokens+1)
        # print(f'Every time start index of boxes: {index}')
        for blk_idx in range(0,len(first_token_ext)-1):
            token_span = first_token_ext[blk_idx+1] - first_token_ext[blk_idx]
            index=json_obj['blocks']['first_token_idx_list'].index(first_token_ext[blk_idx])
            # print(f'extension last index: {index}')
            # exit('++++++++++++++++++')
            bb_blk = json_obj['blocks']['boxes'][index]
            # print(f"number of boxes: {len(json_obj['blocks']['boxes'])}")
            bb_blk[0] = max(0, min(bb_blk[0], width))
            bb_blk[1] = max(0, min(bb_blk[1], height))
            bb_blk[2] = max(0, min(bb_blk[2], width))
            bb_blk[3] = max(0, min(bb_blk[3], height))
            list_blk_bbs.extend([bb_blk for _ in range(token_span)])
            # # line_rank_id
            # return_dict["line_rank_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = line_id
            # line_id += 1
            # # line_rank_inner_id
            # if token_span > 1:
            #     return_dict["line_rank_inner_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = [1] + [2] * (token_span -2) + [3]
        # print(list_blk_bbs)
        print(len(list_blk_bbs))
        # exit('+++++++++++++++++++======')
        print()
        # For [CLS] and [SEP]
        list_tokens = (
            [self.cls_token_id]
            + list_tokens[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(list_bbs) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            list_bbs = [cls_bbs] + [sep_bbs]
            list_blk_bbs = [cls_bbs_blk] + [sep_bbs_blk]
        else:  # len(list_bbs) > 0
            list_bbs = [cls_bbs] + list_bbs[:] + [sep_bbs]
            list_blk_bbs = [cls_bbs_blk] + list_blk_bbs[:] + [sep_bbs_blk]
        # print(f'list of bounding boxes: {len(list_bbs)}')
        len_list_tokens = len(list_tokens)
        len_blocks = len(first_token_idx_list)
        # print(f'after adding cls and sep tokens')
        # print(len_list_tokens)
        # # exit('++++++++++++++++++)))')
        # print(len_blocks)
        # exit('++++++++++++=')
        return_dict["input_ids"][:len_list_tokens] = list_tokens
        return_dict["attention_mask"][:len_list_tokens] = 1
        return_dict["first_token_idxes"][:len(first_token_idx_list)] = first_token_idx_list
        return_dict["block_mask"][:len_blocks] = 1
        return_dict["line_rank_inner_id"] = return_dict["line_rank_inner_id"] * return_dict["attention_mask"]

        bbox_4p_normalized = return_dict["bbox_4p_normalized"]
        bbox_4p_normalized[:len_list_tokens, :] = list_bbs

        # bounding box normalization -> [0, 1]
        bbox_4p_normalized[:, [0, 2, 4, 6]] = bbox_4p_normalized[:, [0, 2, 4, 6]] / width
        bbox_4p_normalized[:, [1, 3, 5, 7]] = bbox_4p_normalized[:, [1, 3, 5, 7]] / height

        if self.backbone_type == "layoutlm":
            bbox_4p_normalized = bbox_4p_normalized[:, [0, 1, 4, 5]]
            bbox_4p_normalized = bbox_4p_normalized * 1000
            bbox_4p_normalized = bbox_4p_normalized.astype(int)

        return_dict["bbox_4p_normalized"] = bbox_4p_normalized
        bbox = return_dict["bbox"]
        print(f'list of block boxes: {len(list_blk_bbs)}')
        print(f'list of tokens: {len_list_tokens}')
        bbox[:len_list_tokens, :] = list_blk_bbs
        # exit('+++++++++++++++++')
        # bbox -> [0, 1000)
        bbox[:, [0, 2]] = bbox[:, [0, 2]] / width * 1000
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / height * 1000
        bbox = bbox.astype(int)
        return_dict["bbox"] = bbox

        st_indices = [
            indices[0]
            for indices in box_to_token_indices
            if indices[0] < self.max_seq_length
        ]
        return_dict["are_box_first_tokens"][st_indices] = True

        # Label for tagging
        # classes_dic = json_obj["parse"]["class"]
        # for class_name in self.class_names:
        #     # if class_name == "O":
        #     #     continue
        #     if class_name not in classes_dic:
        #         continue

        #     for word_list in classes_dic[class_name]:
        #         # At first, connect the class and the first box
        #         is_first, last_word_idx = True, -1
        #         for word_idx in word_list:
        #             if word_idx >= len(box2token_span_map):
        #                 break
        #             box2token_span_start, box2token_span_end = box2token_span_map[
        #                 word_idx
        #             ]
        #             for converted_word_idx in range(
        #                 box2token_span_start, box2token_span_end
        #             ):
        #                 if converted_word_idx >= self.max_seq_length:
        #                     break

        #                 if class_name == 'O':
        #                     return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
        #                         "O"
        #                     ]
        #                     continue
        #                 if is_first:
        #                     return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
        #                         f"B-{class_name}"
        #                     ]
        #                     is_first = False
        #                 else:
        #                     return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
        #                         f"I-{class_name}"
        #                     ]
        return_dict["bio_labels"][0] = -100
        return_dict["bio_labels"][len_list_tokens:] = -100
        # Label for linking
        # relations = json_obj["parse"]["relations"]

        # for relation in relations:
        #     if relation[0] >= len(box2token_span_map) or relation[1] >= len(box2token_span_map):
        #         continue
        #     if (
        #         box2token_span_map[relation[0]][0] >= self.max_seq_length
        #         or box2token_span_map[relation[1]][0] >= self.max_seq_length
        #     ):
        #         continue

        #     word_from = box2token_span_map[relation[0]][0]
        #     word_to = box2token_span_map[relation[1]][0]
        #     return_dict["el_labels_seq"][word_to][word_from] = 1.0
        # return_dict["el_label_seq_mask"][1:len_list_tokens, 1:len_list_tokens] = 1.0
        # B_idx_all = np.array(first_token_idx_list)
        # return_dict["el_labels_blk"][:len(first_token_idx_list), :len(first_token_idx_list)] = \
        #     return_dict["el_labels_seq"][B_idx_all[:, np.newaxis], B_idx_all[np.newaxis, :]]
        # return_dict["el_label_blk_mask"][:len_blocks, :len_blocks] = 1.0

        for k in return_dict.keys():
            if isinstance(return_dict[k], np.ndarray):
                return_dict[k] = torch.from_numpy(return_dict[k])
        # print(return_dict)
        # exit('+++++++++++++++++++++++==')
        return return_dict, len_of_tokens


    def _getitem_geo(self, idx):
        json_obj = self.examples[idx]
        # print(json_obj)
        # exit('++++++++++++++++++')
        track_word_corrds, divide_block= self.__handlingTruncData__(json_obj)
        # track_word_corrds=[]
        # divide_block=[]
        if len(track_word_corrds) ==0:
            print('The tokens less than 512 +++++++++++++++++++++++')
            # exit('+++++++++++++++=')
            return_dict = {}

            width = json_obj["meta"]["imageSize"]["width"]
            height = json_obj["meta"]["imageSize"]["height"]

            img_path = os.path.join(self.dataset_root_path, json_obj["meta"]["image_path"])

            image = cv2.resize(cv2.imread(img_path, 1), (self.img_w, self.img_h))
            image = image.astype("float32").transpose(2, 0, 1)

            return_dict["image_path"] = img_path
            return_dict["image"] = image
            return_dict["size_raw"] = np.array([width, height])

            return_dict["input_ids"] = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
            return_dict["bbox_4p_normalized"] = np.zeros((self.max_seq_length, 8), dtype=np.float32)
            return_dict["attention_mask"] = np.zeros(self.max_seq_length, dtype=int)
            return_dict["first_token_idxes"] = np.zeros(self.max_block_num, dtype=int)
            return_dict["block_mask"] = np.zeros(self.max_block_num, dtype=int)
            return_dict["bbox"] = np.zeros((self.max_seq_length, 4), dtype=np.float32)
            return_dict["line_rank_id"] = np.zeros(self.max_seq_length, dtype="int32")
            return_dict["line_rank_inner_id"] = np.ones(self.max_seq_length, dtype="int32")

            return_dict["are_box_first_tokens"] = np.zeros(self.max_seq_length, dtype=np.bool_)
            return_dict["bio_labels"] = np.zeros(self.max_seq_length, dtype=int)
            return_dict["el_labels_seq"] = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
            return_dict["el_label_seq_mask"] = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
            return_dict["el_labels_blk"] = np.zeros((self.max_block_num, self.max_block_num), dtype=np.float32)
            return_dict["el_label_blk_mask"] = np.zeros((self.max_block_num, self.max_block_num), dtype=np.float32)

            list_tokens = []
            list_bbs = [] # word boxes
            list_blk_bbs = [] # block boxes
            box2token_span_map = []

            box_to_token_indices = []
            cum_token_idx = 0

            cls_bbs = [0.0] * 8
            cls_bbs_blk = [0] * 4
            nest_list_tokens=[]

            for word_idx, word in enumerate(json_obj["words"]):
                this_box_token_indices = []

                tokens = word["tokens"]
                bb = word["boundingBox"]
                if len(tokens) == 0:
                    tokens.append(self.unk_token_id)

                if len(list_tokens) + len(tokens) > self.max_seq_length - 2:
                    print(f"text: {word['text']}")
                    print(f'tokens: {tokens}')
                    print(f'index: {word_idx}')
                    # 
                    
                    
                    break # truncation for long documents

                box2token_span_map.append(
                    [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
                )  # including st_idx, start from 1
                list_tokens += tokens
                # min, max clipping
                for coord_idx in range(4):
                    bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
                    bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

                bb = list(itertools.chain(*bb))
                bbs = [bb for _ in range(len(tokens))]
                # print(bbs)
                # exit('++++++++++++++++++++++++')

                for _ in tokens:
                    cum_token_idx += 1
                    this_box_token_indices.append(cum_token_idx) # start from 1
                # print(f'this box token indices: {this_box_token_indices}')
                # exit('++++++++++++++++++++++')
                list_bbs.extend(bbs)
                box_to_token_indices.append(this_box_token_indices)
            # print(f'box to token indices: {box_to_token_indices}')
            # print(f'len of box to token : {len(box_to_token_indices)}')
            # print(f'num of bbox: {list_bbs}')
            # print(f'len of bbox: {len(list_bbs)}')
            # exit('++++++++++++++++s')
            # print(f'list tokens: {list_tokens}')
            print(len(list_tokens))
            print(f'token bounding box: {len(list_bbs)}')
            # exit('+++++++++++++++++++++')
            sep_bbs = [width, height] * 4
            sep_bbs_blk = [width, height] * 2
            first_token_idx_list = json_obj['blocks']['first_token_idx_list'][:self.max_block_num]
            # print(f"token idx :{first_token_idx_list}")
            print(f'len of token idx: {len(first_token_idx_list)}')
            print(first_token_idx_list[-1])
            # exit('+++++++++++++++++++')
            if first_token_idx_list[-1] > len(list_tokens):
                blk_length = self.max_block_num
                for blk_id, first_token_idx in enumerate(first_token_idx_list):
                    if first_token_idx > len(list_tokens):
                        blk_length = blk_id
                        break
                first_token_idx_list = first_token_idx_list[:blk_length]
                
            first_token_ext = first_token_idx_list + [len(list_tokens) + 1]
            # print(first_token_ext)
            # exit('+++++++++++++++')
            line_id = 1
            for blk_idx in range(len(first_token_ext) - 1):
                token_span = first_token_ext[blk_idx+1] - first_token_ext[blk_idx]
                # block box
                bb_blk = json_obj['blocks']['boxes'][blk_idx]
                bb_blk[0] = max(0, min(bb_blk[0], width))
                bb_blk[1] = max(0, min(bb_blk[1], height))
                bb_blk[2] = max(0, min(bb_blk[2], width))
                bb_blk[3] = max(0, min(bb_blk[3], height))
                list_blk_bbs.extend([bb_blk for _ in range(token_span)])
                # print(list_blk_bbs)
                # exit('+++++++++++++++++')
                # line_rank_id
                return_dict["line_rank_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = line_id
                # print(return_dict['line_rank_id'])
                # exit('++++++++++++++===')
                line_id += 1
                # line_rank_inner_id
                if token_span > 1:
                    return_dict["line_rank_inner_id"][first_token_ext[blk_idx]:first_token_ext[blk_idx+1]] = [1] + [2] * (token_span - 2) + [3]

            # For [CLS] and [SEP]
            list_tokens = (
                [self.cls_token_id]
                + list_tokens[: self.max_seq_length - 2]
                + [self.sep_token_id]
            )
            if len(list_bbs) == 0:
                # When len(json_obj["words"]) == 0 (no OCR result)
                list_bbs = [cls_bbs] + [sep_bbs]
                list_blk_bbs = [cls_bbs_blk] + [sep_bbs_blk]
            else:  # len(list_bbs) > 0
                list_bbs = [cls_bbs] + list_bbs[: self.max_seq_length - 2] + [sep_bbs]
                list_blk_bbs = [cls_bbs_blk] + list_blk_bbs[: self.max_seq_length - 2] + [sep_bbs_blk]

            len_list_tokens = len(list_tokens)
            len_blocks = len(first_token_idx_list)
            # print(len_list_tokens)
            # exit('++++++++++++++++++)))')
            # print(len_blocks)
            # exit('++++++++++++=')
            return_dict["input_ids"][:len_list_tokens] = list_tokens
            return_dict["attention_mask"][:len_list_tokens] = 1
            return_dict["first_token_idxes"][:len(first_token_idx_list)] = first_token_idx_list
            return_dict["block_mask"][:len_blocks] = 1
            return_dict["line_rank_inner_id"] = return_dict["line_rank_inner_id"] * return_dict["attention_mask"]

            bbox_4p_normalized = return_dict["bbox_4p_normalized"]
            bbox_4p_normalized[:len_list_tokens, :] = list_bbs

            # bounding box normalization -> [0, 1]
            bbox_4p_normalized[:, [0, 2, 4, 6]] = bbox_4p_normalized[:, [0, 2, 4, 6]] / width
            bbox_4p_normalized[:, [1, 3, 5, 7]] = bbox_4p_normalized[:, [1, 3, 5, 7]] / height

            if self.backbone_type == "layoutlm":
                bbox_4p_normalized = bbox_4p_normalized[:, [0, 1, 4, 5]]
                bbox_4p_normalized = bbox_4p_normalized * 1000
                bbox_4p_normalized = bbox_4p_normalized.astype(int)

            return_dict["bbox_4p_normalized"] = bbox_4p_normalized
            bbox = return_dict["bbox"]

            bbox[:len_list_tokens, :] = list_blk_bbs
            # bbox -> [0, 1000)
            bbox[:, [0, 2]] = bbox[:, [0, 2]] / width * 1000
            bbox[:, [1, 3]] = bbox[:, [1, 3]] / height * 1000
            bbox = bbox.astype(int)
            return_dict["bbox"] = bbox

            st_indices = [
                indices[0]
                for indices in box_to_token_indices
                if indices[0] < self.max_seq_length
            ]
            return_dict["are_box_first_tokens"][st_indices] = True

            # Label for tagging
            # classes_dic = json_obj["parse"]["class"]
            # for class_name in self.class_names:
            #     # if class_name == "O":
            #     #     continue
            #     if class_name not in classes_dic:
            #         continue

            #     for word_list in classes_dic[class_name]:
            #         # At first, connect the class and the first box
            #         is_first, last_word_idx = True, -1
            #         for word_idx in word_list:
            #             if word_idx >= len(box2token_span_map):
            #                 break
            #             box2token_span_start, box2token_span_end = box2token_span_map[
            #                 word_idx
            #             ]
            #             for converted_word_idx in range(
            #                 box2token_span_start, box2token_span_end
            #             ):
            #                 if converted_word_idx >= self.max_seq_length:
            #                     break

            #                 if class_name == 'O':
            #                     return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
            #                         "O"
            #                     ]
            #                     continue
            #                 if is_first:
            #                     return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
            #                         f"B-{class_name}"
            #                     ]
            #                     is_first = False
            #                 else:
            #                     return_dict["bio_labels"][converted_word_idx] = self.bio_class_idx_dic[
            #                         f"I-{class_name}"
            #                     ]
            return_dict["bio_labels"][0] = -100
            return_dict["bio_labels"][len_list_tokens:] = -100
            # Label for linking
            # relations = json_obj["parse"]["relations"]

            # for relation in relations:
            #     if relation[0] >= len(box2token_span_map) or relation[1] >= len(box2token_span_map):
            #         continue
            #     if (
            #         box2token_span_map[relation[0]][0] >= self.max_seq_length
            #         or box2token_span_map[relation[1]][0] >= self.max_seq_length
            #     ):
            #         continue

            #     word_from = box2token_span_map[relation[0]][0]
            #     word_to = box2token_span_map[relation[1]][0]
            #     return_dict["el_labels_seq"][word_to][word_from] = 1.0
            # return_dict["el_label_seq_mask"][1:len_list_tokens, 1:len_list_tokens] = 1.0
            # B_idx_all = np.array(first_token_idx_list)
            # return_dict["el_labels_blk"][:len(first_token_idx_list), :len(first_token_idx_list)] = \
            #     return_dict["el_labels_seq"][B_idx_all[:, np.newaxis], B_idx_all[np.newaxis, :]]
            # return_dict["el_label_blk_mask"][:len_blocks, :len_blocks] = 1.0

            for k in return_dict.keys():
                if isinstance(return_dict[k], np.ndarray):
                    return_dict[k] = torch.from_numpy(return_dict[k])
            return return_dict
        final_dict=[]
        data_truc= 1
        count=0
        processed_tokens=0
        for words, blocks in zip(track_word_corrds,divide_block):
            count+=1
            return_dict, processed_tokens= self.__preprocess__(json_obj, words, blocks, processed_tokens, count)
            data_truc+=1
            final_dict.append(return_dict)
        return final_dict
        # exit('++++++++++++++++++++++++')
    def __getitem__(self, idx):
        if self.model_head == "vie":
            print(f'idx value:{idx}')
            return_dict = self._getitem_geo(idx)
            print(return_dict)
            exit('++++++++++++++')

        else:
            raise ValueError(f"Unknown self.model_head={self.model_head}")

        return return_dict



