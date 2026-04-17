"""
-----------------------------------------------------------------------------------------
*                              NEWGEN SOFTWARE TECHNOLOGIES LIMITED
*
* Group:Number Theory
* Product/Project: Intelligent Document Processing
* Module:
* File Name: model_inference_utill.py
* Author: Ritwick Roy
* Date (DD/MM/YYYY):
* Description: This file contains Common functions for model inference.
*
* -----------------------------------------------------------------------------------------
*                              CHANGE HISTORY
* -----------------------------------------------------------------------------------------
* Date(DD/MM/YYYY)               Change By              Change Description(Bug No.(If Any))
* -----------------------------------------------------------------------------------------
* 18/06/2020                     Employee name          Description of the changes made
"""

import math
from functools import cmp_to_key
from math import dist

from sklearn.cluster import DBSCAN
from transformers import BertTokenizer

from src.main.common.common_functions import *
from src.main.infer_model import infer_model_constants


def scale_bounding_box(box, width_scale, height_scale):
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale)
    ]


def merge_words_in_bbox(ocr_data, bbox):
    merged_words = []
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    for word_info in ocr_data:
        word_x1, word_y1, word_x2, word_y2 = word_info["x1"], word_info["y1"], word_info["x2"], word_info["y2"]
        if bbox_x1-2 <= word_x1 <= bbox_x2+2 and bbox_x1-2 <= word_x2 <= bbox_x2+2 and \
            bbox_y1-2 <= word_y1 <= bbox_y2+2 and bbox_y1-2 <= word_y2 <= bbox_y2+2:
            merged_words.append(word_info)
    simple_text = ' '.join([x[constants.WORD] for x in merged_words])
    return simple_text


def restructure_result_format(final_result_set, idx2label, ocr_data, test_result, b_box_result, conf_result, img_size):
    w, h = img_size[0], img_size[1]
    for key, value in final_result_set.items():
        if idx2label[key] != 'O':
            text = []
            b_box = []
            confs = []
            for tests in value:
                text_info = merge_words_in_bbox(ocr_data, tests[1].copy())
                box = tests[1]
                b_box = b_box + [box]
                value = text_info if len(text_info) > 0 else tests[0]
                text = text + [value]
                if not isinstance(tests[2], list):
                    confs = confs + [tests[2]]
                else:
                    confs = confs + tests[2]

            test_result.update({idx2label[key]: text})
            b_box_result.update({idx2label[key]: b_box})
            conf_result.update({idx2label[key]: confs})


def get_words_and_normalized_boxes(image, ocr_result):
    words = []
    bbox = []

    width, height = image.size
    width_scale = 1000 / width
    height_scale = 1000 / height

    for ll in ocr_result:
        bbox.append([ll[infer_model_constants.X1], ll[infer_model_constants.Y1], ll[infer_model_constants.X2],
                     ll[infer_model_constants.Y2]])
        words.append(ll[constants.WORD])

    normalized_word_boxes = []
    for box in bbox:
        normalized_word_boxes.append(scale_bounding_box(box, width_scale, height_scale))

    assert len(words) == len(normalized_word_boxes), " Words Length and Normalized words length mismatch"
    return words, normalized_word_boxes


def get_tokens(words, normalized_word_boxes, tokenizer):
    token_boxes = []
    word_tokens = []
    for word, box in zip(words, normalized_word_boxes):
        tokens = tokenizer.tokenize(word)
        word_token = [""]
        for token in tokens:
            if token.startswith(""):
                word_token[-1] = word_token[-1] + token.replace("#", "")
            else:
                word_token.append(token)
        token_boxes.extend([box] * len(word_token))
        word_tokens += word_token
    return token_boxes, word_tokens


def most_common(preds, pred):
    if len(preds) > 0:
        pred = max(set(preds), key=preds.count)
    return pred


def condition_satisfy(curr_box, all_boxes, i, temp_text, sep_index):
    return (curr_box != all_boxes[i] and len(temp_text) > 0) or (i + 1 in sep_index and len(temp_text) > 0)


def get_sep_index(all_text, sep_index):
    if len(all_text) > 512:
        if infer_model_constants.PAD in all_text:
            sep_index = [index-1 for (index, item) in enumerate(all_text) if item == infer_model_constants.PAD]
        else:
            sep_index = [len(all_text) - 2]
    return sep_index


def update_text_and_conf(temp_text, text, temp_preds, pred, conf, temp_confs):
    count = 0
    for j in range(len(temp_text)):
        if temp_preds[j] == pred:
            text += temp_text[j].replace("##", "")
            conf += temp_confs[j]
            count+=1
    return text, conf, count


def handle_edge_case(i, sep_index, temp_values, all_values):
    temp_text, temp_confs, temp_preds, curr_box = temp_values
    all_text, all_confidences, all_predictions, all_boxes = all_values
    if i+1 not in sep_index :
        temp_text.append(all_text[i])
        temp_confs.append(all_confidences[i])
        temp_preds.append(all_predictions[i])
        curr_box = all_boxes[i]
    return curr_box


def create_annotation_data(all_text, all_boxes, all_predictions, results, all_confidences, results_conf, idx2label):
    label2idx = {v:k for k,v in idx2label.items()}
    results_bbox, results_text, results_pred = results
    curr_box: list = []
    temp_preds: list = []
    temp_confs: list = []
    temp_text: list = []
    sep_index: list = [i for i in range(len(all_text)) if all_text[i] == infer_model_constants.SEP]

    for i in range(len(all_text)):
        if all_text[i] not in [infer_model_constants.CLS, infer_model_constants.SEP, infer_model_constants.PAD]:
            if condition_satisfy(curr_box, all_boxes, i, temp_text, sep_index):
                if i+1 in sep_index:
                    temp_text.append(all_text[i])
                    temp_confs.append(all_confidences[i])
                    temp_preds.append(all_predictions[i])
                    curr_box = all_boxes[i]
                text = ""
                pred = label2idx["O"]
                preds = [x for x in temp_preds if idx2label[x] != 'O']
                conf = 0
                
                pred = most_common(preds, pred)
                text, conf, count = update_text_and_conf(temp_text, text, temp_preds, pred, conf, temp_confs)
                conf = float(np.round(conf * 100 / count, 2))
                results_text.append(text)
                results_conf.append(conf)
                results_pred.append(pred)
                results_bbox.append(curr_box)

                temp_text = []
                temp_confs = []
                temp_preds = []
                curr_box = handle_edge_case(i, sep_index, [temp_text, temp_confs, temp_preds, curr_box],
                                            [all_text, all_confidences, all_predictions, all_boxes])
            elif curr_box == all_boxes[i]:
                temp_text.append(all_text[i])
                temp_confs.append(all_confidences[i])
                temp_preds.append(all_predictions[i])
            elif len(temp_text) == 0:
                temp_text.append(all_text[i])
                temp_confs.append(all_confidences[i])
                temp_preds.append(all_predictions[i])
                curr_box = all_boxes[i]


def contour_sort(a, b):
    if abs(a[1][1] - b[1][1]) <= 15:
        return a[1][0] - b[1][0]
    return a[1][1] - b[1][1]


def update_result_set(k, final_result_set, result_set, alpha):
    texts = [x[0] for x in result_set[k]]
    bboxes = [x[1] for x in result_set[k]]
    confs = [x[2] for x in result_set[k]]
    avg_w = np.mean([abs(x[0] - x[2]) for x in bboxes])
    avg_h = np.mean([abs(x[1] - x[3]) for x in bboxes])
    eps = np.sqrt(avg_w ** 2 + avg_h ** 2) * alpha
    clustering = DBSCAN(eps=eps, min_samples=1).fit(bboxes)
    label_set = set(clustering.labels_)
    for l in label_set:
        selected = list(np.where(clustering.labels_ == l)[0])
        selected_texts = [x for i, x in enumerate(texts) if i in selected]
        selected_boxes = [x for i, x in enumerate(bboxes) if i in selected]
        selected_confs = [x for i, x in enumerate(confs) if i in selected]
        text_boxes = [[x, y] for x, y in zip(selected_texts, selected_boxes)]
        text_boxes = sorted(text_boxes, key=cmp_to_key(contour_sort))
        text_result = ""
        for tb in text_boxes:
            if text_result == "":
                text_result += tb[0]
            else:
                text_result += " " + tb[0]
        x1 = min([x[0] for x in selected_boxes])
        x2 = max([x[2] for x in selected_boxes])
        y1 = min([x[1] for x in selected_boxes])
        y2 = max([x[3] for x in selected_boxes])
        box_result = [x1, y1, x2, y2]
        conf_result = float(np.round(np.mean(selected_confs), 2))
        if k not in list(final_result_set.keys()):
            final_result_set[k] = []
        final_result_set[k].append([text_result, box_result, [conf_result]])

    return final_result_set


def create_final_resul_for_single_line_label(k, result_set, final_result_set):
    if len(result_set[k]) > 1:
        texts = [x[0] for x in result_set[k]]
        bboxes = [x[1] for x in result_set[k]]
        confs = [x[2] for x in result_set[k]]
        for i, value in enumerate(zip(texts, bboxes, confs)):
            text, bbox, conf = value
            if k not in list(final_result_set.keys()):
                final_result_set[k] = []
            final_result_set[k].append([text, bbox, [conf]])
    else:
        if k not in list(final_result_set.keys()):
            final_result_set[k] = []
        final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], [result_set[k][0][2]]])


def create_final_result(result_set, single_text_labels, idx2label):
    final_result_set = {}
    for k in list(result_set.keys()):
        if idx2label[k] not in single_text_labels:
            alpha = 1.6
            if len(result_set[k]) > 1:
                final_result_set = update_result_set(k, final_result_set, result_set, alpha)
            else:
                if k not in list(final_result_set.keys()):
                    final_result_set[k] = []
                final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], [result_set[k][0][2]]])
        else:
            create_final_resul_for_single_line_label(k, result_set, final_result_set)
        
    return final_result_set


def minimum_distance(bb1, bb2):
    min_distance = 9999999999
    p_11 = np.array((bb1[0], bb1[1]))
    p_12 = np.array((bb1[0], bb1[3]))
    p_13 = np.array((bb1[2], bb1[3]))
    p_14 = np.array((bb1[2], bb1[1]))
    all_points_bb1 = [p_11, p_12, p_13, p_14]
    p_21 = np.array((bb2[0], bb2[1]))
    p_22 = np.array((bb2[0], bb2[3]))
    p_23 = np.array((bb2[2], bb2[3]))
    p_24 = np.array((bb2[2], bb2[1]))
    all_points_bb2 = [p_21, p_22, p_23, p_24]
    for point1 in all_points_bb1:
        for point2 in all_points_bb2:
            dist = abs(np.linalg.norm(point1 - point2))
            if dist < min_distance:
                min_distance = dist
    return min_distance


def get_iou_new(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def model_output_sum(key, box, model_output):
    all_values = model_output[key]
    all_values = sorted(all_values, key=cmp_to_key(contour_sort))
    all_text = ""
    for value in all_values:
        try:
            iou = get_iou_new(value[1], box)
        except Exception as e:
            print(e)
            continue
        if iou > 0:
            if all_text == "":
                all_text = value[0]
            else:
                all_text = all_text + " " + value[0]
    return all_text


def minimum_distance_vertical(bb1, bb2):
    # Calculate the minimum vertical distance between two bounding boxes
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    min_distance_y = min(abs(y1_bb2 - y2_bb1), abs(y1_bb1 - y2_bb2))

    return min_distance_y


def get_iou(bb1, bb2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes (horizontal intersection)
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    x_left = max(x1_bb1, x1_bb2)
    y_top = max(y1_bb1, y1_bb2)
    x_right = min(x2_bb1, x2_bb2)
    y_bottom = min(y2_bb1, y2_bb2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    bb1_area = (x2_bb1 - x1_bb1) * (y2_bb1 - y1_bb1)
    bb2_area = (x2_bb2 - x1_bb2) * (y2_bb2 - y1_bb2)
    union_area = bb1_area + bb2_area - intersection_area

    return intersection_area / union_area


def get_intersection_percentage(bb1, bb2):
    # Calculate the percentage of vertical intersection between two bounding boxes
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    x_left = max(x1_bb1, x1_bb2)
    y_top = max(y1_bb1, y1_bb2)
    x_right = min(x2_bb1, x2_bb2)
    y_bottom = min(y2_bb1, y2_bb2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = max(0, y_bottom - y_top)
    bb1_area = y2_bb1 - y1_bb1
    bb2_area = y2_bb2 - y1_bb2

    return intersection_area / min(bb1_area, bb2_area)


def special_chr_check(bb_token, flag):
    if ',' in bb_token:
        flag = False
    if '-' in bb_token:
        flag = False
    return flag


def area(coordinates):
    length = coordinates[2] - coordinates[0]
    height = coordinates[3] - coordinates[1]
    return length * height


def update_data(i, bounding_box_values, threshold_values, iou_values, other_values, length):
    bb1_height, bb1_width, bb2_height, bb2_width, bb1, bb2 = bounding_box_values
    min_dist_horizontal, min_dist_vertical, eps, inter_percentage = threshold_values
    iou_horizontal, iou_vertical = iou_values
    key, model_output, confs, all_values = other_values

    if (bb1_height >= bb1_width and bb2_height >= bb2_width) or (
            bb1_height <= bb1_width and bb2_height <= bb2_width):
        # This case is when both the boxes are either horizontal or vertically alligned on a document.
        if (min_dist_horizontal <= eps or iou_horizontal > 0 or inter_percentage > 0) or (
                min_dist_vertical <= eps or iou_vertical > 0 or inter_percentage):
            x_left = min(bb1[0], bb2[0])
            y_top = min(bb1[1], bb2[1])
            x_right = max(bb1[2], bb2[2])
            y_bottom = max(bb1[3], bb2[3])
            box = [x_left, y_top, x_right, y_bottom]
            text = model_output_sum(key, box, model_output)
            avg_confs = (confs[0][0] * area(bb1) + confs[1][0] * area(bb2)) / (area(bb1) + area(bb2))
            new_value = [text, box, [avg_confs]]
            all_values.remove(all_values[i])
            all_values.remove(all_values[i])
            all_values.insert(i, new_value)
            length = len(all_values)
        else:
            i = i + 1
    else:
        i = i + 1
    return i, all_values, length


def update_data_for_multi_line_label(all_values, eps, key, model_output, length):
    i = 0
    while i in range(length - 1):
        bb1 = all_values[i][1]
        bb2 = all_values[i + 1][1]
        bb1_token = all_values[i][0]
        bb2_token = all_values[i + 1][0]
        confs = [all_values[i][2], all_values[i + 1][2]]
        min_dist_horizontal = minimum_distance(bb1, bb2)
        min_dist_vertical = minimum_distance_vertical(bb1, bb2)

        try:
            iou_horizontal = get_iou(bb1, bb2)
            iou_vertical = get_iou(bb1, bb2)
            inter_percentage = get_intersection_percentage(bb1, bb2)
        except ZeroDivisionError:
            i = i + 1
            continue
        bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1
        bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2
        bb1_width = bb1_x2 - bb1_x1
        bb1_height = bb1_y2 - bb1_y1
        bb2_width = bb2_x2 - bb2_x1
        bb2_height = bb2_y2 - bb2_y1

        flag1 = True
        flag2 = True
        flag1 = special_chr_check(bb1_token, flag1)
        flag2 = special_chr_check(bb2_token, flag2)

        if len(bb1_token) == 1 or (len(bb1_token) < 3 and flag1 == False):
            temp = bb1_width
            bb1_width = bb1_height
            bb1_height = temp
        if len(bb2_token) == 1 or (len(bb2_token) < 3 and flag2 == False):
            temp = bb2_width
            bb2_width = bb2_height
            bb2_height = temp

        i, all_values, length = update_data(i, [bb1_height, bb1_width, bb2_height, bb2_width, bb1, bb2],
                                    [min_dist_horizontal, min_dist_vertical, eps, inter_percentage]
                                    , [iou_horizontal, iou_vertical], [key, model_output, confs, all_values], length)
        if length == 1:
            break

    return all_values


def check_vertical_distribution(bb1, bb2):
    y1 = bb1[1]
    y2 = bb2[1]
    return abs(y1 - y2)


def calculate_merging_values(i, all_values, v_eps, h_eps, key, model_output, length):
    while i in range(length - 1):
        bb1 = all_values[i][1]
        bb2 = all_values[i + 1][1]
        confs = [all_values[i][2], all_values[i + 1][2]]
        vertical_distance = check_vertical_distribution(bb1, bb2)
        hori_distance = abs(bb1[2] - bb2[0])
        try:
            IOU = get_iou_new(bb1, bb2)
        except (ZeroDivisionError, AssertionError):
            i = i + 1
            continue
        if (vertical_distance <= v_eps and hori_distance < h_eps) or IOU > 0.1:
            x_left = min(bb1[0], bb2[0])
            y_top = min(bb1[1], bb2[1])
            x_right = max(bb1[2], bb2[2])
            y_bottom = max(bb1[3], bb2[3])
            box = [x_left, y_top, x_right, y_bottom]
            text = model_output_sum(key, box, model_output)
            avg_confs = [(confs[0][0] * area(bb1) + confs[1][0] * area(bb2)) / (area(bb1) + area(bb2))]
            new_value = [text, box, avg_confs]
            all_values.remove(all_values[i])
            all_values.remove(all_values[i])
            all_values.insert(i, new_value)
            length = len(all_values)
            if length == 1:
                break
        else:
            i = i + 1


def calculate_eps(w, h):
    if w > h:
        v_eps = round(h * 1.5 / 100)  # 10round(number)
        h_eps = round(w * 5 / 100)  # 36
    else:
        v_eps = round(h * 1.1 / 100)  # 10round(number)
        h_eps = round(w * 5.8 / 100)  # 36
    return v_eps, h_eps


def merge_surrounding(data, model_output, w, h, vertical_merge_labels, idx2label):
    for key in list(data.keys()):
        if idx2label[key] in vertical_merge_labels:
            eps = 100
            all_values = data[key]
            length = len(all_values)
            if length > 1:
                update_data_for_multi_line_label(all_values, eps, key, model_output, length)
            else:
                continue
        else:
            v_eps, h_eps = calculate_eps(w, h)
            all_values = data[key]
            length = len(all_values)
            if length > 1:
                i = 0
                calculate_merging_values(i, all_values, v_eps, h_eps, key, model_output, length)
            else:
                continue


def update_es_data(common_dict, status_dict):
    current_es_data = get_data_from_es(es_settings=common_dict, es_type=constants.PIPELINE_RUNNING_STATUS)
    save_data_in_es(es_settings=common_dict, final_json=status_dict,
                    es_type=constants.PIPELINE_RUNNING_STATUS)

    for hit in current_es_data[constants.HITS][constants.HITS]:
        if hit[constants.SOURCE].get(constants.TIMESTAMP) == common_dict[constants.TIMESTAMP]:
            current_id = hit['_id']
            delete_data_from_es(common_dict, current_id)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, class_idx2name):
    m = outputs[infer_model_constants.PRED_LOGITS].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs[infer_model_constants.PRED_BOXES].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == infer_model_constants.NO_OBJECT:
            objects.append({constants.LABEL_DATA: class_label, infer_model_constants.SCORES: float(score),
                            infer_model_constants.BBOXES: [float(elem) for elem in bbox]})

    return objects


def prepare_result_for_table_transformer(original_results):
    final_result = {constants.LABEL_DATA: [], infer_model_constants.BBOXES: [], 'scores': []}
    for result in original_results:
        final_result[constants.LABEL_DATA].append(result[constants.LABEL_DATA])
        final_result[infer_model_constants.BBOXES].append(result[infer_model_constants.BBOXES])
        final_result[infer_model_constants.SCORES].append(result[infer_model_constants.SCORES])
    return final_result


idx2label_for_table_transformer = {
    0: infer_model_constants.TABLE,
    1: infer_model_constants.TABLE_COLUMN,
    2: infer_model_constants.TABLE_ROW,
    3: infer_model_constants.TABLE_COLUMN_HEADER,
    4: infer_model_constants.TABLE_PROJECTED_ROW_HEADER,
    5: infer_model_constants.TABLE_SPANNING_CELL,
    6: infer_model_constants.NO_OBJECT
}


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def parse_str_from_seq(seq, box_first_token_mask, bio_class_names):
    seq = seq[box_first_token_mask]
    res_str_list = []
    for i, label_id_tensor in enumerate(seq):
        label_id = label_id_tensor.item()
        if label_id < 0:
            raise ValueError("The label of words must not be negative!")
        res_str_list.append(bio_class_names[label_id])

    return res_str_list


def get_class_names(dataset_root_path):
    class_names_file = os.path.join(dataset_root_path, infer_model_constants.CLASS_NAME_FILE)
    class_names = (
        open(class_names_file, "r", encoding="utf-8").read().strip().split("\n")
    )
    return class_names


def optimize_output(bboxes, labels, texts):
    new_bboxes = []
    new_label = []
    new_text = []
    text = ''
    for i, bbox in enumerate(bboxes):
        if bbox in new_bboxes:
            text += texts[i]
        else:
            if i > 0:
                new_text.append(text)
            text = ''
            new_bboxes.append(bbox)
            new_label.append(labels[i])
    new_text.append(text)
    return new_bboxes, new_label, new_text


def prepare_bio_class(idx2label):
    idx_to_label_list = list(idx2label.values())
    print(idx_to_label_list)
    bio_class_names = ["O"]
    for class_name in idx_to_label_list:
        if not class_name.startswith('O'):
            bio_class_names.extend([f"B-{class_name}", f"I-{class_name}"])
    return bio_class_names


def parse_prediction(input_data, pr_labels, idx2label):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    bio_class_name_list = prepare_bio_class(idx2label)

    bio_class_names = {
        infer_model_constants.BIO_CLASS_NAME: bio_class_name_list}
    res_dict = []

    pr_str_i = parse_str_from_seq(
        pr_labels,
        input_data[infer_model_constants.ARE_BOX_FIRST_TOKENS],
        bio_class_names[infer_model_constants.BIO_CLASS_NAME],
    )

    for key in input_data:
        if key != constants.IMAGE_NAME:
            input_data[key] = input_data[key].squeeze(0)
    box_first_token_mask = input_data[infer_model_constants.ARE_BOX_FIRST_TOKENS].cpu().tolist()
    num_valid_tokens = input_data[infer_model_constants.ATTENTION_MASK].sum().item()

    input_ids = input_data[infer_model_constants.INPUT_IDS].cpu().tolist()

    width, height = input_data[infer_model_constants.SIZE_RAW].cpu().tolist()
    block_boxes = input_data[infer_model_constants.BBOX].float()
    block_boxes[:, [0, 2]] = block_boxes[:, [0, 2]] / 1000 * width
    block_boxes[:, [1, 3]] = block_boxes[:, [1, 3]] / 1000 * height
    block_boxes = block_boxes.to(torch.long).cpu().tolist()

    for token_idx in range(num_valid_tokens):
        if box_first_token_mask[token_idx]:
            valid_idx = sum(box_first_token_mask[:token_idx + 1]) - 1

            # add word info
            ids = [input_ids[token_idx]]
            tok_tmp_idx = token_idx + 1
            while tok_tmp_idx < num_valid_tokens and not box_first_token_mask[tok_tmp_idx]:
                ids.append(input_ids[tok_tmp_idx])
                tok_tmp_idx += 1
            word = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))

            # add coord info
            block_box = block_boxes[token_idx]
            res_dict.append({
                infer_model_constants.TOKEN_ID: token_idx,
                infer_model_constants.PRED_KEY: pr_str_i[valid_idx],
                constants.TEXT: word,
                infer_model_constants.COORDS: block_box
            })

    bboxes = []
    label = []
    text = []
    for result in res_dict:
        bboxes.append(result[infer_model_constants.COORDS])
        label.append(result[infer_model_constants.PRED_KEY])
        text.append(result[constants.TEXT])
    bboxes, label, text = optimize_output(bboxes, label, text)
    return {infer_model_constants.KEYS_EXTRACTION: {}, infer_model_constants.KEYS_BBOXES: bboxes,
            constants.LABEL_DATA: label, constants.TEXT: text}


def get_font_size(coords):
    for coord in coords:
        coord.update({infer_model_constants.BBOX: [coord['x1'], coord['y1'], coord['x2'], coord['y2']]})

    coords = sort_coords(coords, axis=1)
    font_size = round(np.mean([coord['y2'] - coord['y1'] for coord in coords]))

    return font_size, coords


def sort_coords(coords, axis=None):
    if axis == 1:
        points = [[item['bbox'][1], item['bbox'][3]] for item in coords]
    elif axis == 0:
        points = [[item['bbox'][0], item['bbox'][2]] for item in coords]
    else:
        points = [item['bbox'][:2] for item in coords]
    points_dist = []
    for coord in points:
        points_dist.append(round(dist((0, 0), coord), 2))
    sorted_order = np.argsort(points_dist)
    sorted_coords = [coords[i] for i in sorted_order]
    return sorted_coords


def get_lines(coords):
    font_size, coords = get_font_size(coords)
    line_num = 0
    lines = {}
    for i in range(len(coords) - 1):
        word_dist = coords[i + 1][infer_model_constants.BBOX][1] - coords[i][infer_model_constants.BBOX][1]
        if word_dist <= font_size + 1:
            if not line_num in lines:
                lines.update({line_num: [coords[i]]})
            else:
                lines[line_num].append(coords[i])
        else:
            if not line_num in lines:
                lines.update({line_num: [coords[i]]})
            else:
                lines[line_num].append(coords[i])
            line_num += 1
            # lines.update({line_num: [coords[i+1]]})
    lines[len(lines) - 1].append(coords[i + 1])
    for item in lines:
        x_min = min([x[infer_model_constants.BBOX][0] for x in lines[item]])
        x_max = max([x[infer_model_constants.BBOX][2] for x in lines[item]])

        y_min = min([y[infer_model_constants.BBOX][1] for y in lines[item]])
        y_max = max([y[infer_model_constants.BBOX][3] for y in lines[item]])
        lines[item] = {
            infer_model_constants.BLOCK_BBOX: [x_min, y_min, x_max, y_max],
            constants.WORDS: lines[item]
        }

    return lines


def update_prepare_data_struct(prepare_data_struct, blocks, tokenizer):
    num_tokens = 0

    for block in blocks:
        real_word_idx = 0
        prepare_data_struct[constants.BLOCKS][infer_model_constants.BOXES].append(
            blocks[block][infer_model_constants.BLOCK_BBOX])

        class_seq = []
        for coord in blocks[block][constants.WORDS]:
            word_text = coord[constants.WORD]

            bb = coord[infer_model_constants.BBOX]
            bb = [bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))

            word_obj = {constants.TEXT: word_text, infer_model_constants.TOKENS: tokens,
                        infer_model_constants.BOUNDING_BOX: bb}
            prepare_data_struct[constants.WORDS].append(word_obj)

            if real_word_idx == 0:
                prepare_data_struct[constants.BLOCKS][infer_model_constants.FIRST_TOKEN_IDX_LIST].append(num_tokens + 1)
            num_tokens += len(tokens)

            class_seq.append(len(prepare_data_struct[constants.WORDS]) - 1)  # word index

            real_word_idx += 1

        if infer_model_constants.BLOCK_LABEL in blocks[block]:
            if blocks[block][infer_model_constants.BLOCK_LABEL] == infer_model_constants.TABLE_ROW:
                label = infer_model_constants.DATA_CELL
            elif blocks[block][infer_model_constants.BLOCK_LABEL] == infer_model_constants.TABLE_COLUMN_HEADER:
                label = infer_model_constants.HEADER_CELL
            elif blocks[block][infer_model_constants.BLOCK_LABEL] == infer_model_constants.TRASH:
                label = infer_model_constants.TRASH
            else:
                label = 'O'

            prepare_data_struct[infer_model_constants.PARSE][infer_model_constants.CLASS][label].append(class_seq)


def prepare_ocr_data(ocr_data, image):
    voca = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(voca, do_lower_case=True)
    classes = ['data_cell', 'header_cell', 'trash', 'O']

    blocks = get_lines(ocr_data)

    if blocks is not None:
        prepare_data_struct = {
            infer_model_constants.META: {
                infer_model_constants.IMAGE_PATH: str,
                infer_model_constants.IMAGESIZE: {
                    infer_model_constants.WIDTH: int,
                    infer_model_constants.HEIGHT: int
                },
                infer_model_constants.VOCA: str
            },
            constants.BLOCKS: {
                infer_model_constants.FIRST_TOKEN_IDX_LIST: [],
                infer_model_constants.BOXES: [],
            },
            constants.WORDS: [],
            infer_model_constants.PARSE: {
                infer_model_constants.CLASS: {},
                infer_model_constants.RELATIONS: []
            }
        }

        for c in classes:
            prepare_data_struct[infer_model_constants.PARSE][infer_model_constants.CLASS].update({c: []})

        image_h, image_w = image.shape[:2]

        prepare_data_struct[infer_model_constants.META][infer_model_constants.IMAGESIZE][
            infer_model_constants.WIDTH] = image_w
        prepare_data_struct[infer_model_constants.META][infer_model_constants.IMAGESIZE][
            infer_model_constants.HEIGHT] = image_h
        prepare_data_struct[infer_model_constants.META][infer_model_constants.VOCA] = voca

        update_prepare_data_struct(prepare_data_struct, blocks, tokenizer)

    return prepare_data_struct


def get_final_conf(predictions):
    confidences = []
    for prediction in predictions:
        confs = []
        for pred in prediction:
            confs.append(pred.max().item())
        confidences += confs
    return confidences


def create_local_model_path(common_dict, model_name):
    dt_object = datetime.fromtimestamp(int(common_dict[constants.TIMESTAMP]) / 1000.0,
                                       tz=pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H_%M_%S")
    model_folder = f"../../../../../{model_name}"
    model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_path = f"{model_folder}/{common_dict[constants.PIPELINE_ID]}_" \
                 f"{common_dict[constants.VERSION]}_{dt_object}"
    return model_path
