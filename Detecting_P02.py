import json
import os.path
import cv2
import argparse
import numpy as np
import tqdm
import glob
from utils.rule import *
from object_association.object_association import Object_Association
from post_processing_for_tracking.track_object import tracker

CLASSES = ["motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet", "P2Helmet", "P2NoHelmet", "P0Helmet",
           "P0NoHelmet"]


def detecting_p0(video_folder, detect_results, head_results, threshold=0.14):
    obj_association = Object_Association(video_folder=video_folder,
                                         display=False, head_thresh=threshold,
                                         prediction_path=detect_results,
                                         head_label_path=head_results)
    video_p0 = []
    for video_path in tqdm.tqdm(sorted(glob.glob(video_folder + "/*"))):
        cap = cv2.VideoCapture(video_path)
        tracking = tracker.my_tracking()
        frame_id = 0
        video_id = int(video_path.split('/')[-1].split('.')[0])
        is_P0 = False
        while cap.isOpened():
            try:
                _, frame = cap.read()
                _, _, _ = frame.shape
            except Exception as e:
                break
            frame_id += 1
            results_obj_association = obj_association.foward_frame(frame, video_id, frame_id)
            bbox_motor = []
            for rs in results_obj_association:
                box = rs.get_box_info()
                bbox_motor.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])
            if len(bbox_motor) > 0:
                output_vehicle, is_P2, is_P0 = tracking.update(np.array(bbox_motor), results_obj_association)
            if is_P0:
                video_p0.append(video_id)
                break
    return video_p0


def detecting_p2(video_folder, detect_results, head_results, threshold=0.23):
    obj_association = Object_Association(video_folder=video_folder,
                                         display=False, head_thresh=threshold,
                                         prediction_path=detect_results,
                                         head_label_path=head_results)
    video_p2 = []
    for video_path in tqdm.tqdm(sorted(glob.glob(video_folder + "/*"))):
        cap = cv2.VideoCapture(video_path)
        tracking = tracker.my_tracking()
        frame_id = 0
        video_id = int(video_path.split('/')[-1].split('.')[0])
        is_P2 = False
        while cap.isOpened():
            try:
                _, frame = cap.read()
                _, _, _ = frame.shape
            except Exception as e:
                break
            frame_id += 1
            results_obj_association = obj_association.foward_frame(frame, video_id, frame_id)
            bbox_motor = []
            for rs in results_obj_association:
                box = rs.get_box_info()
                bbox_motor.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])
            if len(bbox_motor) > 0:
                output_vehicle, is_P2, is_P0 = tracking.update(np.array(bbox_motor), results_obj_association)
            if is_P2:
                video_p2.append(video_id)
                break
    return video_p2
