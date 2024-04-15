from ensemble_boxes import weighted_boxes_fusion


def perform_weighted_boxes_fusion(
        pred_confs_models,
        pred_boxes_models,
        pred_classes_models,
        resolution_dict,
        weights=None,
        IOU_THRESH=0.5,
        CONF_THRESH=None,
        FINAL_CONF_THRESH=1e-3):
    wbf_boxes_dict = dict()
    wbf_scores_dict = dict()
    wbf_labels_dict = dict()
    for image_id, res in resolution_dict.items():
        res_array = np.array([1920, 1080, 1920, 1080])  # [W, H, W, H]
        all_model_boxes = []
        all_model_scores = []
        all_model_classes = []
        for boxes, scores, classes in zip(pred_boxes_models, pred_confs_models, pred_classes_models):
            if len(boxes[image_id]) < 1:
                pred_boxes_norm = []
                scores_model = []
                classes_model = []
                # continue
            else:
                pred_boxes_norm = (boxes[image_id] / res_array).clip(min=0., max=1.)
                # print(pred_boxes_norm)
                # exit(0)
                scores_model = scores[image_id]
                classes_model = classes[image_id]

            all_model_boxes.append(pred_boxes_norm)
            all_model_scores.append(scores_model)
            all_model_classes.append(classes_model)
        # if len(all_model_boxes) < 1:
        #     wbf_boxes_dict[image_id] = None
        #     wbf_scores_dict[image_id] = None
        #     wbf_labels_dict[image_id] = None
        #     continue
        # Perform weighted box fusion.
        boxes, scores, labels = weighted_boxes_fusion(
            all_model_boxes,
            all_model_scores,
            all_model_classes,
            weights=weights,
            iou_thr=IOU_THRESH,
            skip_box_thr=CONF_THRESH, conf_type="avg")  # avg - box_and_model_avg - absent_model_aware_avg - max
        final_boxes = (boxes * res_array)
        final_boxes = final_boxes.astype("int")
        # Box cordinates in [xmin, ymin, width, height] in de-normalized form.
        final_boxes[:, 2:] = final_boxes[:, 2:] - final_boxes[:, :2]
        wbf_boxes_dict[image_id] = final_boxes.tolist()
        wbf_scores_dict[image_id] = np.round(scores, 5).tolist()
        wbf_labels_dict[image_id] = labels.tolist()

    return wbf_boxes_dict, wbf_scores_dict, wbf_labels_dict

