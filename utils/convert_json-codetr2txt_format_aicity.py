import glob
import json
import os.path
import cv2
from pycocotools.coco import COCO
import numpy as np
import tqdm
import argparse


def json_to_dict(obj, key):
    _dict = {}
    if type(obj) is dict:
        for key in obj.keys():
            _dict[key] = json_to_dict(obj[key], key)
        return _dict
    if type(obj) is list:
        for idx, child_obj in enumerate(obj):
            _dict[f"{key}{idx}"] = json_to_dict(child_obj, key)
        return _dict
    return obj


def sorter(item):
    item = item.split(",")
    return int(item[1])


def sorter1(item):
    item = item.split(",")
    return int(item[0])


classes = ["motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet", "P2Helmet", "P2NoHelmet", "P0Helmet",
           "P0NoHelmet"]

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--images_id', required=False, default="",
                     help='file image id')
    arg.add_argument('--image_bbox', default="", help="output bbox")
    arg.add_argument('--output', default="", help="output bbox format txt aicity")

    args = arg.parse_args()
    image_id = args.images_id
    image_bbox = args.image_bbox
    filename_output = args.output
    f = open(image_id)
    json_label_data = json.load(f)

    with open(image_bbox, 'r') as f:
        read_data = f.read()
        for idx, product in enumerate(json.loads(read_data)):
            data = json_to_dict(product, idx)
            data["id"] = idx
            json_label_data["annotations"].append(data)

    with open("merge_bbox_id.json", 'w') as f:
        json.dump(json_label_data, f)
    annFile = "merge_bbox_id.json"
    coco = COCO(annFile)
    catIds = coco.getCatIds()
    print(catIds)
    imgIds = coco.getImgIds()
    print(len(imgIds))
    cats = coco.loadCats(coco.getCatIds())
    print(cats)
    detect_file1 = open(filename_output, "w")
    output_file_detect1 = []
    c = 0
    for ids in range(0, len(imgIds)):
        img = coco.loadImgs(imgIds[ids])[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        name = img["file_name"]
        name_base = os.path.splitext(name)[0]
        st_video = int(name_base.split("_")[0])
        st_frame = int(name_base.split("_")[1])
        for an in anns:
            score = an["score"]
            classid = an["category_id"]
            xmin, ymin, w, h = int(an["bbox"]["bbox0"]), int(an["bbox"]["bbox1"]), int(an["bbox"]["bbox2"]), int(
                an["bbox"]["bbox3"])
            xmin = int(np.clip(xmin, 1, 1919))
            ymin = int(np.clip(ymin, 1, 1079))
            w = int(np.clip(w, 1, 1919))
            h = int(np.clip(h, 1, 1079))
            output_file_detect1.append("%d,%d,%d,%d,%d,%d,%d,%.4f" %
                                       (st_video, st_frame, xmin, ymin, w, h, classid + 1, score
                                        ))
    output_file_detect1 = sorted(output_file_detect1, key=sorter)
    output_file_detect1 = sorted(output_file_detect1, key=sorter1)
    detect_file1.write("\n".join(np.array(output_file_detect1)))
    print("final")
