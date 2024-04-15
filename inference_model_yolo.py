import glob
import os.path

import tqdm
from PIL import Image
from ultralytics import YOLO
from utils.utils import *
import argparse
from utils.yolov7_onnx import init_model

class_name = ["motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet", "P2Helmet", "P2NoHelmet", "P0Helmet",
              "P0NoHelmet"]


def get_data_resolution(image_dir_path):
    image_res = dict()

    for img_file in image_dir_path:
        image_id = os.path.splitext(os.path.basename(img_file))[0]
        image = Image.open(img_file)
        image_res[image_id] = image.size

    return image_res


def get_predictions_yolov8(model, image_filenames, conf_thres=0.01, type_p=False):
    preds_scores = dict()
    pred_boxes = dict()
    pred_classes = dict()

    for image_file in tqdm.tqdm(image_filenames):
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        if type_p:
            pred_results = model.predict(image_file, augment=False, imgsz=1280, conf=conf_thres, verbose=False, iou=0.6)[
                0].boxes.cpu()
        else:
            pred_results = model.predict(image_file, augment=True, imgsz=832, conf=conf_thres, verbose=False, iou=0.6)[
                0].boxes.cpu()
        preds_scores[image_id] = pred_results.conf.unsqueeze(dim=1).numpy()
        pred_boxes[image_id] = pred_results.xyxy.numpy()
        pred_classes[image_id] = pred_results.cls.int().unsqueeze(dim=1).numpy()
    return preds_scores, pred_boxes, pred_classes


def get_predictions_yolov7(w, image_filenames):
    preds_scores = dict()
    pred_boxes = dict()
    pred_classes = dict()
    model = init_model(w)
    for image_file in tqdm.tqdm(image_filenames):
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        image = cv2.imread(image_file)
        bboxes, class_ids, scores = model.infer(image)

        preds_scores[image_id] = scores
        pred_boxes[image_id] = bboxes
        pred_classes[image_id] = class_ids

    return preds_scores, pred_boxes, pred_classes


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--video-frame-folder', required=False,
                     help='video-frame-folder',
                     default="/vnpt_dev_project/code/vnpt_challenge/aicity2024/flow/aicity2024_track5_test/track5_video_test_frame/")
    arg.add_argument('--models-dir', default="training/weights/", help="model dir")
    args = arg.parse_args()
    SAMPLE_IMG_DIR = args.video_frame_folder
    MODELS_DIR = args.models_dir
    mode = args.mode_post
    save_image = args.save_image
    model_dict_yolov8 = dict()
    model_yolov7 = ""

    sample_img_files = sorted(glob.glob(SAMPLE_IMG_DIR + "*/*"))
    data_res_dict = get_data_resolution(sample_img_files)

    ckpt_files = glob.glob(MODELS_DIR + "/*/weights/best.*")
    #### yolov8
    print("Using model:  \n")
    for pt_file in ckpt_files:
        basename = pt_file.split("/")[-3]
        if "yolov8" in basename:
            model_dict_yolov8[basename] = YOLO(pt_file)
            print("\t {}  \n".format(basename))
        else:
            model_dict_yolov7 = pt_file
            print("\t {}  \n".format(basename))

    for j, model in enumerate(model_dict_yolov8):
        modelname = model
        model_obj = model_dict_yolov8[model].value()
        confs_scores, box_preds, cls_preds = get_predictions_yolov8(model_obj, sample_img_files)
        detect_file = open("results_detection/{}.txt".format(modelname), "w")
        output_file_detect = []
        for i, (img_file, box_data, confs, classid) in enumerate(
                zip(sample_img_files, box_preds.values(), confs_scores.values(), cls_preds.values())):
            st_video = int(img_file.split("/")[-2])
            st_frame = int(os.path.splitext(os.path.basename(img_file))[0].split("_")[1])
            for i, object in enumerate(box_data):
                xmin = int(np.clip(object[0], 1, 1919))
                ymin = int(np.clip(object[1], 1, 1079))
                w = int(np.clip(object[2] - object[0], 1, 1919))
                h = int(np.clip(object[3] - object[1], 1, 1079))
                output_file_detect.append("%d,%d,%d,%d,%d,%d,%d,%.6f" %
                                          (st_video, st_frame, xmin, ymin, w, h, classid[i] + 1,
                                           confs[i]
                                           ))
        detect_file.write("\n".join(np.array(output_file_detect)))
    #### yolov7
    pred_confs = []
    pred_boxes = []
    pred_classes = []
    scores_dict = []
    confs_scores_v7, box_preds_v7, cls_preds_v7 = get_predictions_yolov7(model_yolov7, sample_img_files)
    detect_file = open("results_detection/{}.txt".format(model_yolov7.split(".")[0]), "w")
    output_file_detect = []
    for i, (img_file, box_data, confs, classid) in enumerate(
            zip(sample_img_files, box_preds_v7.values(), confs_scores_v7.values(), cls_preds_v7.values())):
        st_video = int(img_file.split("/")[-2])
        st_frame = int(os.path.splitext(os.path.basename(img_file))[0].split("_")[1])
        for i, object in enumerate(box_data):
            xmin = int(np.clip(object[0], 1, 1919))
            ymin = int(np.clip(object[1], 1, 1079))
            w = int(np.clip(object[2] - object[0], 1, 1919))
            h = int(np.clip(object[3] - object[1], 1, 1079))
            output_file_detect.append("%d,%d,%d,%d,%d,%d,%d,%.6f" %
                                      (st_video, st_frame, xmin, ymin, w, h, classid[i] + 1,
                                       confs[i]
                                       ))

    detect_file.write("\n".join(np.array(output_file_detect)))
