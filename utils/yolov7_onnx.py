# Inference for ONNX model
import glob

import cv2

cuda = True

import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict, namedtuple
import torch
import torch.nn.functional as F
import math


def letterbox(im, new_shape=(1280, 1280), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


class init_model():
    def __init__(self, w):
        providers = ['CUDAExecutionProvider']
        self.session = ort.InferenceSession(w, providers=providers)
        self.names = ["motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet", "P2Helmet", "P2NoHelmet",
                      "P0Helmet",
                      "P0NoHelmet"]
        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(self.names)}
        self.outname = [i.name for i in self.session.get_outputs()]
        self.inname = [i.name for i in self.session.get_inputs()]

    def infer(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        inp = {self.inname[0]: im}
        outputs = self.session.run(self.outname, inp)[0]
        bboxes = []
        class_ids = []
        scores = []
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 3)
            bboxes.append(box)
            class_ids.append(cls_id)
            scores.append(score)
        return bboxes, class_ids, scores


    def infer_test(self, image):
        frame = image.copy()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        inp = {self.inname[0]: im}
        outputs = self.session.run(self.outname, inp)[0]
        ori_images = [img.copy()]
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 3)
            name = self.names[cls_id]
            color = self.colors[name]
            name += ' ' + str(score)
            cv2.rectangle(frame, box[:2], box[2:], color, 2)
            cv2.putText(frame, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
        return frame


if __name__ == '__main__':
    w = "../model/yolov7/best-d6.onnx"
    model = init_model(w)
    for path in glob.glob("data/test/images1/*"):
        frame = cv2.imread(path)
        rs = model.infer_test(frame)
        cv2.imshow("xxx", cv2.resize(rs, dsize=(1440, 736)))
        cv2.waitKey()
