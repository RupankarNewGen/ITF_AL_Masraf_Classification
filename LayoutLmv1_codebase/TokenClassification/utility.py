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

import torch
import os
from torchvision.utils import draw_bounding_boxes
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import cv2
import logging
from typing import List, Tuple
from configparser import ConfigParser


# config snippet
configur = ConfigParser()
conf_folder_path: str = "src/main/extraction/"
config_file_name: str = "/home/ng6309/datascience/hridesh/layoutlmv1/extraction/train_valid.ini"

# if os.path.exists(f"{conf_folder_path}/{config_file_name}"):
#     configur.read(f"{conf_folder_path}/{config_file_name}")

configur.read(config_file_name)

def get_gpu_memory_usage() -> None:
	print(torch.cuda.get_device_name(0))
	print('Memory Usage:')
	print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
	print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')


def data_convert(dataset: List = None) -> Tuple[List, List, List]:
    if dataset is None:
        return [], [], []

    words = []
    boxes = []
    labels = []
    image_name = []
    for datas in dataset:
        words.append(datas['words'])
        boxes.append(datas['bbox'])
        labels.append(datas['labels'])
        image_name.append(datas["filename"])
    return image_name,words, boxes, labels


def normalize(points: list, width: int, height: int) -> list:
    x0, y0, x2, y2 = [int(p) for p in points]
    val = int(configur['PARAMS']['norm_val'])
    zero_val = int(configur['PARAMS']['nill_val'])
    x0 = int(val * (x0 / width))
    x2 = int(val * (x2 / width))
    y0 = int(val * (y0 / height))
    y2 = int(val * (y2 / height))
    if x0 > val:
        x0 = val
    if x0 < zero_val:
        x0 = zero_val
    if x2 > val:
        x2 = val
    if x2 < zero_val:
        x2 = zero_val
    if y0 > val:
        y0 = val
    if y0 < zero_val:
        y0 = zero_val
    if y2 > val:
        y2 = val
    if y2 < zero_val:
        y2 = zero_val
    return [x0, y0, x2, y2]


def plot_bbox(img, boxes):
	drawn_boxes = draw_bounding_boxes(img, boxes, colors="red")
	show(drawn_boxes)


def bbox_inhouse(image_path, boxes):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	print(image.shape)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	print(boxes)
	for box in boxes:
		print(box[0], box[1])
		image = cv2.rectangle(image, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][0])), (0, 255, 0), 2)
	plt.imshow(image)
	plt.show()


def show(imgs):
	if not isinstance(imgs, list):
		imgs = [imgs]
	fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
	for i, img in enumerate(imgs):
		img = img.detach()
		img = F.to_pil_image(img)
		axs[0, i].imshow(np.asarray(img))
		axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def set_basic_config_for_logging(full_folder_path: str = None, 
                                 filename: str = None):
    """
    Set the basic config for logging python program.
    :return: None
    """

    # full_folder_path = os.path.join(os.getcwd(), folder_path)

    # Create the folder if it doesn't exist
    os.makedirs(full_folder_path, exist_ok=True)

    # If the filename is not provided, use a default filename based on the current timestamp
    # if filename is None:
    #     filename = f"data_preparation_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]}.log"

    # Get the full path for the log file
    full_file_path = os.path.join(full_folder_path, filename)

    # Configure logging
    logging.basicConfig(
        filename=full_file_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Example log message
    logging.info("Logging started")


def get_logger_object_and_setting_the_loglevel(log_level: str = "info"):
    """
    get the logger object and set the loglevel for the logger object
    :return: Logger Object
    """
    # Creating an object
    logger_object = logging.getLogger()
    level = None

    if log_level == "INFO":
        level = logging.INFO
    elif log_level == "DEBUG":
        level = logging.DEBUG
    elif log_level == "CRITICAL":
        level = logging.CRITICAL
    else:
        level = logging.ERROR

    # Setting the threshold of logger to DEBUG
    logger_object.setLevel(level)
    return logger_object
