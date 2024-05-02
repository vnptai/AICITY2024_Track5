import os
import glob
import cv2
import numpy as np
import glob
import os
from Detecting_P02 import detecting_p0, detecting_p2
from PIL import Image
import tqdm
from utils.utils import perform_weighted_boxes_fusion


def sorter(item):
    item = item.split(",")
    return int(item[1])


def sorter1(item):
    item = item.split(",")
    return int(item[0])


def get_data_resolution(image_dir_path):
    image_res = dict()

    for img_file in image_dir_path:
        image_id = os.path.splitext(os.path.basename(img_file))[0]
        image = Image.open(img_file)
        image_res[image_id] = image.size

    return image_res


def get_original_box(path, order='class_last'):
    original_bbox_info = {}
    for line in path:
        data = line.split(',')
        video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, class_confidence = int(data[0]), int(
            data[1]), \
            int(data[2]), int(data[3]), int(data[4]), int(data[5]), int(data[6]), float(data[7].split('\n')[0])
        if video_id not in original_bbox_info.keys():
            original_bbox_info[video_id] = {}
        if frame not in original_bbox_info[video_id].keys():
            original_bbox_info[video_id][frame] = []
        if order == 'class_last':
            original_bbox_info[video_id][frame].append(
                [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height, class_confidence, class_id - 1])
        else:
            original_bbox_info[video_id][frame].append(
                [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height, class_id - 1, class_confidence])
    return original_bbox_info


def ensemble_model_wbf(sample_img_files, models, weights, iou_thresh, conf_thresh, final_conf_thresh=0.00001,
                       save_path=""):
    pred_confs = []
    pred_boxess = []
    pred_classess = []
    for index in models:
        aicity_labels = models[index]
        preds_scores = dict()
        pred_boxes = dict()
        pred_classes = dict()
        for frame_path in sample_img_files:
            image_id = os.path.splitext(os.path.basename(frame_path))[0]
            video_id = int(frame_path.split("/")[-2])
            frame_id = int(os.path.splitext(os.path.basename(frame_path))[0].split("_")[1])
            frame_rs = []
            if video_id in aicity_labels.keys():
                if frame_id in aicity_labels[video_id].keys():
                    for frame_object in aicity_labels[video_id][frame_id]:
                        frame_rs.append(frame_object)
            if len(frame_rs) < 1:
                preds_scores[image_id] = []
                pred_boxes[image_id] = []
                pred_classes[image_id] = []
                continue
            frame_rs = np.array(frame_rs)
            preds_scores[image_id] = frame_rs[:, 5]
            pred_boxes[image_id] = frame_rs[:, :4]
            pred_classes[image_id] = frame_rs[:, 4]
        pred_confs.append(preds_scores)
        pred_boxess.append(pred_boxes)
        pred_classess.append(pred_classes)
    ###############
    boxes_dict_wbf, scores_dict_wbf, labels_dict_wbf = perform_weighted_boxes_fusion(
        pred_confs,
        pred_boxess,
        pred_classess,
        data_res_dict,
        weights=weights,
        IOU_THRESH=iou_thresh,
        CONF_THRESH=conf_thresh,
        FINAL_CONF_THRESH=final_conf_thresh)

    output_file_detect = []
    for i, (img_file, box_data, confs, classid) in enumerate(
            zip(sample_img_files, boxes_dict_wbf.values(), scores_dict_wbf.values(), labels_dict_wbf.values())):
        video_id = int(img_file.split("/")[-2])
        frame_id = int(os.path.splitext(os.path.basename(img_file))[0].split("_")[1])
        if box_data is None:
            continue
        for i, object in enumerate(box_data):
            xmin = int(np.clip(object[0], 1, 1919))
            ymin = int(np.clip(object[1], 1, 1079))
            w = int(np.clip(object[2], 1, 1919))
            h = int(np.clip(object[3], 1, 1079))
            output_file_detect.append("%d,%d,%d,%d,%d,%d,%d,%.6f" %
                                      (video_id, frame_id, xmin, ymin, w, h, classid[i] + 1, confs[i]
                                       ))

    file_output = open(save_path, "w")
    file_output.write("\n".join(np.array(output_file_detect)))
    return output_file_detect


def score_correction_module(output_rs_wbf, video_P0, video_P2):
    output_score_correction = []
    for da in output_rs_wbf:
        data = da.split(",")
        video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, class_confidence = int(data[0]), int(data[1]), \
            int(data[2]), int(data[3]), int(data[4]), int(data[5]), int(data[6]), float(data[7].split('\n')[0])
        xmin, ymin, w, h = bb_left, bb_top, bb_width, bb_height
        xmin = int(np.clip(xmin, 1, 1919))
        ymin = int(np.clip(ymin, 1, 1079))
        w = int(np.clip(w, 1, 1919))
        h = int(np.clip(h, 1, 1079))
        if video_id in video_P0:
            if class_id in [8, 9]:
                class_confidence = class_confidence + 0.1
                class_confidence = 0.99 if class_confidence > 0.99 else class_confidence
        if video_id in video_P2:
            if class_id in [6, 7]:
                class_confidence = class_confidence + 0.2
                class_confidence = 0.99 if class_confidence > 0.99 else class_confidence
        output_score_correction.append("%d,%d,%d,%d,%d,%d,%d,%.6f" %
                                       (video_id, frame, xmin, ymin, w, h, class_id, class_confidence
                                        ))
    return output_score_correction


if __name__ == '__main__':
    print("Loading file results......")
    SAMPLE_IMG_DIR = "./training/aicity_dataset/test2024/frame/*/*"
    sample_img_files = sorted(glob.glob(SAMPLE_IMG_DIR))
    data_res_dict = get_data_resolution(sample_img_files)
    model2 = open("results_detection/codetr-data_v2_16.txt", "r").readlines()
    model3 = open("results_detection/codetr_data_v2_2.txt", "r").readlines()
    model4 = open("results_detection/codetr_data_v1_16.txt", "r").readlines()
    model5 = open("results_detection/codetr_data_v1_2.txt", "r").readlines()
    model6 = open("results_detection/codetr_data_v1_8.txt", "r").readlines()
    model7 = open("results_detection/yolov8x_data_v2_1.txt", "r").readlines()
    model8 = open("results_detection/yolov8x_data_v1.txt", "r").readlines()
    model9 = open("results_detection/yolov7-d6_data_v1.txt", "r").readlines()
    model10 = open("results_detection/yolov8x-p2_data_v1.txt", "r").readlines()
    model11 = open("results_detection/yolov8x-p6_data_v1.txt", "r").readlines()
    model12 = open("results_detection/yolov8x_data_v2_2.txt", "r").readlines()
    aicity_labels2 = get_original_box(model2, order='conf_last')
    aicity_labels3 = get_original_box(model3, order='conf_last')
    aicity_labels4 = get_original_box(model4, order='conf_last')
    aicity_labels5 = get_original_box(model5, order='conf_last')
    aicity_labels6 = get_original_box(model6, order='conf_last')
    aicity_labels7 = get_original_box(model7, order='conf_last')
    aicity_labels8 = get_original_box(model8, order='conf_last')
    aicity_labels9 = get_original_box(model9, order='conf_last')
    aicity_labels10 = get_original_box(model10, order='conf_last')
    aicity_labels11 = get_original_box(model11, order='conf_last')
    aicity_labels12 = get_original_box(model12, order='conf_last')
    #### step 1 Ensemble w3
    list_models4 = {"model1": aicity_labels5,
                    "model2": aicity_labels6, "model3": aicity_labels3, "model4": aicity_labels4}
    weights4 = [1, 1, 3, 2]
    iou_thresh = 0.7
    conf_thresh = 0.05
    models = list_models4
    save_path = "results_detection/w4.txt"
    if os.path.exists(save_path):
        print("skip wbs step 1....")
    else:
        print("WBF step 1....")
        output_rs_wbf_w12 = ensemble_model_wbf(sample_img_files, models, weights4, iou_thresh, conf_thresh,
                                               save_path=save_path)

    model1 = open("results_detection/w4.txt", "r").readlines()
    aicity_labels1 = get_original_box(model1, order='conf_last')

    #### Step 2 Ensemble w12
    list_models12 = {"model1": aicity_labels1, "model2": aicity_labels2,
                     "model3": aicity_labels3, "model4": aicity_labels4, "model5": aicity_labels5,
                     "model6": aicity_labels6, "model7": aicity_labels7, "model8": aicity_labels8,
                     "model9": aicity_labels9,
                     "model10": aicity_labels10, "model11": aicity_labels11, "model12": aicity_labels12}
    weights12 = [12, 10, 10, 7, 7, 7, 1, 1, 1, 1, 1, 1]
    iou_thresh = 0.66
    conf_thresh = 0.01
    models = list_models12
    save_path = "results_detection/w12.txt"
    if os.path.exists(save_path):
        print("skip wbs step 2....")
    else:
        print("WBF step 2....")
        output_rs_wbf_w12 = ensemble_model_wbf(sample_img_files, models, weights12, iou_thresh, conf_thresh,
                                               save_path=save_path)
    #### Step 3 Ensemble w13
    model0 = open("results_detection/w12.txt", "r").readlines()
    aicity_labels13 = get_original_box(model0, order='conf_last')
    list_models13 = {"model0": aicity_labels13, "model1": aicity_labels1,
                     "model2": aicity_labels2,
                     "model3": aicity_labels3, "model4": aicity_labels4, "model5": aicity_labels5,
                     "model6": aicity_labels6, "model7": aicity_labels7, "model8": aicity_labels8,
                     "model9": aicity_labels9,
                     "model10": aicity_labels10, "model11": aicity_labels11, "model12": aicity_labels12,
                     }
    weights13 = [13, 12, 10, 10, 7, 7, 7, 1, 1, 1, 1, 1, 1]
    iou_thresh = 0.7
    conf_thresh = 0.01
    models = list_models13
    save_path = "results_detection/w13_final_ensemble.txt"
    print("WBF step 3....")
    output_rs_wbf_13 = ensemble_model_wbf(sample_img_files, models, weights13, iou_thresh, conf_thresh,
                                          save_path=save_path)
    #### Step 4 - Detecting P0 P2
    video_folder = "./training/aicity_dataset/test2024/videos/"
    detect_results = save_path
    head_results = "results_head/codet-head.txt"
    print("Detecting P2....")
    video_P2 = detecting_p2(video_folder, detect_results, head_results)
    print("Detecting P2 : ", video_P2)
    print("Detecting P0....")
    video_P0 = detecting_p0(video_folder, detect_results, head_results)
    print("Detecting P0 : ", video_P0)
    #### Step 4 - Score correction module
    print("Score Correction....")
    output_final_submit = score_correction_module(output_rs_wbf_13, video_P0, video_P2)
    #### Save final result submit
    output_final_submit = sorted(output_final_submit, key=sorter)
    output_final_submit = sorted(output_final_submit, key=sorter1)
    file_final_submit = open("final_submit.txt", "w")
    file_final_submit.write("\n".join(np.array(output_final_submit)))
    print("Final....")
