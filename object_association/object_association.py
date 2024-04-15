import os
import cv2
import argparse
from glob import glob
import numpy as np

from object_association.utils import *
from object_association.detection_object import Motor, Human, Head

MOTOR_CLASS = ['motorbike']
DRIVER_CLASSES = ['DHelmet', 'DNoHelmet']
P1_CLASSES = ['P1Helmet', 'P1NoHelmet']
P2_CLASSES = ['P2Helmet', 'P2NoHelmet']
P0_CLASSES = ['P0Helmet', 'P0NoHelmet']

HUMAN_CLASSES = DRIVER_CLASSES + P1_CLASSES + P2_CLASSES + P0_CLASSES

OVERLAP_60_RULE = False
REMOVE_SMALL_OBJS_RULE = True
REMOVE_MOTOR_WIHOUT_DRIVER = False
DISPLAY = True
WRITE_TO_VIDEO = False
MIN_SIZE = 40

if WRITE_TO_VIDEO:
    os.makedirs(WRITE_TO_VIDEO, exist_ok=True)


def write_enembled_results(file, objects, video_id, frame_id, w, h, conf_thres=0.5):
    if not len(objects):
        return

    for obj in objects:
        left, top, right, bottom, class_id, confidence, _ = obj.get_box_info()
        if left <= 1 and top <= 1 and right <= 1 and bottom <= 1:
            left, top, right, bottom = left * w, top * h, right * w, bottom * h
        left, top, right, bottom, class_id = int(left), int(top), int(right), int(bottom), int(class_id)

        width = right - left
        height = bottom - top
        if confidence < conf_thres:
            continue
        line = ','.join(
            [str(i) for i in [video_id, frame_id, left, top, width, height, class_id + 1, confidence]]) + '\n'
        file.write(line)


class Object_Association(object):
    def __init__(self, prediction_path='result_files/best.txt',
                 head_label_path='result_files/head.txt',
                 video_folder=None,
                 export_path=None,
                 write_to_video=WRITE_TO_VIDEO,
                 display=DISPLAY,
                 remove_small_objs_rule=REMOVE_SMALL_OBJS_RULE,
                 overlap_60_rule=OVERLAP_60_RULE,
                 remove_motor_without_driver=REMOVE_MOTOR_WIHOUT_DRIVER,
                 head_thresh=0.12,
                 conf_thres=0.2,
                 head_motor_overlap_thresh=0.75):
        self.head_motor_overlap_thresh = head_motor_overlap_thresh
        self.remove_small_objs_rule = remove_small_objs_rule
        self.overlap_60_rule = overlap_60_rule
        self.remove_motor_without_driver = remove_motor_without_driver
        self.head_thresh = head_thresh
        self.conf_thres = conf_thres

        self.video_folder = video_folder
        self.write_to_video = write_to_video
        self.display = display
        self.w = 1920
        self.h = 1080

        self.acity_labels = get_original_box(prediction_path, order='conf_last')
        self.heads = get_original_box(head_label_path, order='conf_last', head=True)

        self.export_file = open(export_path, 'w+') if export_path else None

    def draw_box(self, image, objects, classes=class_dict, show_label=True, show_combined=False, box_only=False):
        """Draw bounding box

        Args:
            image (np array): opencv image
            objects (list): list of objects (type: Motor, Human, Head)
        """
        if box_only:
            for bb_left, bb_top, bb_right, bb_bottom in objects:
                cv2.rectangle(image, (int(bb_left), int(bb_top)), (int(bb_right), int(bb_bottom)), (0, 0, 255), 2)
        else:
            for object in objects:
                bb_left, bb_top, bb_right, bb_bottom, class_id, conf, _ = object.get_box_info()
                label = classes[int(class_id)]
                color = COLORS[int(class_id)]
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(image, (int(bb_left), int(bb_top)), (int(bb_right), int(bb_bottom)), color, 2)
                if object.type == 'motor' and show_combined:
                    expand_bb_left, expand_bb_top, expand_bb_right, expand_bb_bottom = object.combined_box
                    cv2.rectangle(image, (int(expand_bb_left), int(expand_bb_top)),
                                  (int(expand_bb_right), int(expand_bb_bottom)), color, 2)
                if show_label:
                    cv2.rectangle(image, (int(bb_left), int(bb_top)),
                                  (int(bb_left + t_size[0] + 3), int(bb_top + t_size[1] + 4)), color, -1)
                    cv2.putText(image, label, (int(bb_left), int(bb_top + t_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, 2,
                                [255, 255, 255], 2)

    def foward_videos(self):
        for video_path in glob(os.path.join(self.video_folder, '*.mp4')):
            video_id = int(video_path.split('/')[-1].split('.')[0])
            print('================== {} =================='.format(video_id))
            if not os.path.exists(video_path):
                print('Video {} is not exist'.format(video_path))
                continue

            if self.write_to_video:
                video_writer = VideoWriter(os.path.join(WRITE_TO_VIDEO, video_path.split('/')[-1]))

            frame_id = 1
            video_cap = cv2.VideoCapture(video_path)
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if ret == True:
                    org_frame = frame.copy()
                    self.h, self.w, c = frame.shape

                    ### Ensemble frame
                    results, frame = self.foward_frame(frame, video_id=video_id, frame_id=frame_id)

                    if self.write_to_video:
                        video_writer.write(frame)

                    ### Write ensembled bboxes to text file
                    if self.export_file:
                        write_enembled_results(self.export_file, results, video_id, frame_id, self.w, self.h,
                                               conf_thres=self.conf_thres)

                    frame_id += 1

                    if self.display:
                        cv2.imshow('frame', cv2.resize(frame, (1024, 768)))
                        if cv2.waitKey(0) == ord('q'):
                            break
                else:
                    break

            video_cap.release()
            if self.write_to_video:
                video_writer.destroy()
            if self.display:
                cv2.destroyAllWindows()
        if self.export_file:
            self.export_file.close()

    def foward_frame(self, frame, video_id, frame_id):
        frame_heads = []
        frame_humans = []
        frame_motors = []

        if video_id in self.heads.keys():
            if frame_id in self.heads[video_id].keys():
                for head in self.heads[video_id][frame_id]:
                    if head[-1] > self.head_thresh:
                        frame_heads.append(Head(bbox=head))
        if video_id in self.acity_labels.keys():
            if frame_id in self.acity_labels[video_id].keys():
                for frame_object in self.acity_labels[video_id][frame_id]:
                    if frame_object[-2] == 0:
                        if frame_object[-1] > 0.2:
                            frame_motors.append(Motor(bbox=frame_object))
                    else:
                        if frame_object[-2] in [1, 2]:
                            if frame_object[-1] >= 0.15:
                                frame_humans.append(Human(bbox=frame_object))
                        if frame_object[-2] in [3, 4]:
                            if frame_object[-1] >= 0.1:
                                frame_humans.append(Human(bbox=frame_object))
                        if frame_object[-2] in [5, 7]:
                            if frame_object[-1] >= 0.9:
                                frame_humans.append(Human(bbox=frame_object))
                        if frame_object[-2] in [6, 8]:
                            if frame_object[-1] >= 0.1:
                                frame_humans.append(Human(bbox=frame_object))

            ### Attach head to correspond human
            for human in frame_humans:
                human.attach_motor_id(frame_motors)
                human.attach_head_id(frame_heads)

            for head in frame_heads:
                head.attach_motor_id(frame_motors, self.head_motor_overlap_thresh)

            ### Visulize bbounding boxes ###
            if self.display:
                self.draw_box(image=frame, objects=frame_humans, classes=class_dict)
                self.draw_box(image=frame, objects=frame_motors, classes=class_dict, show_combined=False)
                for motor in frame_motors:
                    self.draw_box(image=frame, objects=motor.heads, classes=['head'], show_label=False, box_only=False)

        return frame_motors


if __name__ == '__main__':
    infer = Infer(video_folder=VIDEO_FOLDER,
                  head_thresh=0.2,
                  display=True,
                  prediction_path='best.txt',
                  head_label_path='head.txt')
    infer.foward_videos()
