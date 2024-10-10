import copy
import random
import shutil

import numpy as np
import os
from tqdm import tqdm
import torch
from typing import Tuple, List, Dict, Optional, Union
from collections import OrderedDict
from matplotlib import pyplot as plt
from torchvision.models.detection import *
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers, _mobilenet_extractor
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection._utils import overwrite_eps
import torch.nn.functional as F
from torch import nn
import cv2

def sort_by_area(detect_person_res):
    area_ls = []
    for i, single_res in enumerate(detect_person_res):
        temp_area = []
        s = single_res[2] * single_res[3]
        temp_area.append(s)
        temp_area.append(i)
        area_ls.append(temp_area)
        pass
    sorted_id_ls = np.array(sorted(area_ls, reverse=True))[:, 1]
    return sorted_id_ls

def is_overlap(box1, box2):
    start_x_1, start_y_1, end_x_1, end_y_1 = box1
    start_x_2, start_y_2, end_x_2, end_y_2 = box2
    return not ((start_x_1 > end_x_2) or (start_x_2 > end_x_1) or (start_y_1 > end_y_2) or (start_y_2 > end_y_1))


def divid_person_group(detect_person_res):
    deal_person_detect_res = detect_person_res[:, :4]

    def combine_over_lap(deal_person_detect_res):
        sorted_id_ls = sort_by_area(deal_person_detect_res)
        overlap_group = []

        for i in sorted_id_ls:
            single_res = deal_person_detect_res[int(i)]

            if len(overlap_group) < 1:
                overlap_group.append(single_res)
            else:
                res_is_overlap = False
                for group_id, group_single_res in enumerate(overlap_group):
                    if is_overlap(single_res, group_single_res):
                        res_is_overlap = True
                        overlap_group[group_id][:2] = np.minimum(group_single_res[:2], single_res[:2])
                        overlap_group[group_id][2:4] = np.maximum(group_single_res[2:4], single_res[2:4])

                        break
                if not res_is_overlap:
                    overlap_group.append(single_res)
        return overlap_group

    first_overlap_group = combine_over_lap(deal_person_detect_res)
    return np.array(first_overlap_group)


def cal_overlap_only(box1, box2, x1y1x2y2=False):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    area1 = w1 * h1
    area2 = w2 * h2
    if cw <= 0 or ch <= 0:
        carea = 0
    else:
        carea = cw * ch

    return max(carea / area1, carea / area2)


class fjn_faster_rcnn():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thres = 0.8
        self.nms_thresh = 0.7       
        self.box_nms_thresh = 0.5   
        self.init_model()
        self.box_detections_per_img = 100

        self.overlap_thresh = 0.4
        pass

    def init_model(self):
        model = fasterrcnn_resnet50_fpn(pretrained=True, rpn_nms_thresh=self.nms_thresh, box_nms_thresh=self.box_nms_thresh).to(self.device)
        model.rpn.min_size=30
        model.eval()
        self.detector = model
        self.names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.names))]
        random.seed(1)
        self.bbox_colors = random.sample(colors, len(self.names))
        pass

    def detect_pre_data(self, img_cv):
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_cv_rgb.transpose(2, 0, 1)).float().div(255.0).to(self.device)
        return [img_tensor]

    def detect_single_img_cv(self, img_cv, score_thresh=None):
        img_tensor = self.detect_pre_data(img_cv)

        original_image_sizes: List[Tuple[int, int]] = []
        for img in img_tensor:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.detector.transform(img_tensor, None)

        features = self.detector.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.detector.rpn(images, features, targets)


        detections, detector_losses = self.detector.roi_heads(features, proposals, images.image_sizes, targets)


        detections = self.detector.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        output_dict = detections[0]

        detect_res = []
        if len(output_dict["boxes"]) > 0:
            out_boxes = output_dict["boxes"].cpu()
            out_scores = output_dict["scores"].cpu()
            out_labels = output_dict["labels"].cpu()

            if not score_thresh:
                score_thresh = self.thres

            for idx in range(0, min(out_boxes.shape[0], self.box_detections_per_img)):
                score = out_scores[idx].detach().numpy()
                bbox = out_boxes[idx].detach().numpy()
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cls_pred = out_labels[idx].detach().numpy()

                if score >= score_thresh:
                    if cls_pred == 1:
                        high_overlap = False
                        for single_boxes in detect_res:
                            if single_boxes[-1] == 1:
                                tmp_overlap = cal_overlap_only([x1, y1, x2, y2], single_boxes[:4], x1y1x2y2=True)
                                if tmp_overlap >=0.8:
                                    high_overlap = True
                                    break
                        if not high_overlap:
                            detect_res.append([x1, y1, x2, y2, score, cls_pred])
                    else:
                        detect_res.append([x1, y1, x2, y2, score, cls_pred])
        if len(detect_res)<1:
            detect_res = [[0, 0, 1, 1, 1.0, -1]]
        return np.array(detect_res)

    
    def detect_single_img_tensor(self, img_tensor, score_thresh=None):

        original_image_sizes: List[Tuple[int, int]] = []
        for img in img_tensor:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.detector.transform(img_tensor, None)

        features = self.detector.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.detector.rpn(images, features, targets)


        detections, detector_losses = self.detector.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.detector.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        output_dict = detections[0]

        detect_res = []
        if len(output_dict["boxes"]) > 0:
            out_boxes = output_dict["boxes"].cpu()
            out_scores = output_dict["scores"].cpu()
            out_labels = output_dict["labels"].cpu()

            if not score_thresh:
                score_thresh = self.thres

            for idx in range(0, min(out_boxes.shape[0], self.box_detections_per_img)):
                score = out_scores[idx].detach().numpy()
                bbox = out_boxes[idx].detach().numpy()
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cls_pred = out_labels[idx].detach().numpy()

                if score >= score_thresh:
                    if cls_pred == 1:
                        high_overlap = False
                        for single_boxes in detect_res:
                            if single_boxes[-1] == 1:
                                tmp_overlap = cal_overlap_only([x1, y1, x2, y2], single_boxes[:4], x1y1x2y2=True)
                                if tmp_overlap >=0.8:
                                    high_overlap = True
                                    break
                        if not high_overlap:
                            detect_res.append([x1, y1, x2, y2, score, cls_pred])
                    else:
                        detect_res.append([x1, y1, x2, y2, score, cls_pred])
        if len(detect_res)<1:
            detect_res = [[0, 0, 1, 1, 1.0, -1]]
        return np.array(detect_res)


    def draw_detect_single_image(self, img_cv):
        img_original_cv = copy.deepcopy(img_cv)
        output_dict = self.detect_single_img_cv(img_cv)

        return self.draw_image_with_boxes(img_original_cv, output_dict)

    def draw_image_with_boxes(self, img_cv, bboxes):
        img_original_cv = copy.deepcopy(img_cv)
        if len(bboxes)>0:
            for single_res in bboxes:
                x1, y1, x2, y2, score, cls_pred = single_res
                if cls_pred == -1:
                    continue
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = self.names[int(cls_pred)]

                color = self.bbox_colors[int(cls_pred)]
                color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

                cv2.rectangle(img_original_cv, (x1, y1), (x2, y2), color, thickness=1)

                cv2.rectangle(img_original_cv, (x1 - 2, y1), (x2 + 2, y1 + 19), color, thickness=-1)
                cv2.putText(img_original_cv, class_name + " {:.2f}".format(score), (x1, y1 + 13), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

        return img_original_cv

    def detect_fold_imgs(self, imgs_path, save_path):
        os.makedirs(save_path, exist_ok=True)

        t = tqdm(total=len(os.listdir(imgs_path)), ascii=True)
        for img_file in os.listdir(imgs_path):
            if img_file.lower().endswith('.jpg') or img_file.lower().endswith('.jpeg') or img_file.lower().endswith('.png'):
                img = cv2.imread(os.path.join(imgs_path, img_file), 1)

                res = self.draw_detect_single_image(img)
                output_path = os.path.join(save_path, img_file)
                cv2.imwrite(output_path, res)

                t.set_postfix({f'detect image': img_file})
                t.update(1)
        t.close()
        pass

    def get_label(self, imgs_path, save_txt_path):
        os.makedirs(save_txt_path, exist_ok=True)

        for img_file in os.listdir(imgs_path):
            if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                name = os.path.splitext(img_file)[0]

                input_imgs = cv2.imread(os.path.join(imgs_path, img_file), 1)
                boxes = self.detect_single_img_cv(input_imgs)

                textfile = open(os.path.join(save_txt_path, name + '.txt'), 'w+')

                for item in boxes:
                    cx, cy = (item[0] + item[2]) / 2 / input_imgs.shape[1], (item[1] + item[3]) / 2 / input_imgs.shape[0]
                    width, height = (item[2] - item[0]) / input_imgs.shape[1], (item[3] - item[1]) / input_imgs.shape[0]
                    textfile.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(item[-1].item(), cx, cy, width, height))
                    
                textfile.close()
        pass

    def detect_for_hidden_person(self, img_cv, interest_score_thresh=0.075, interest_cls=1):
        img_tensor = self.detect_pre_data(img_cv)

        original_image_sizes: List[Tuple[int, int]] = []
        for img in img_tensor:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.detector.transform(img_tensor, None)

        features = self.detector.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, proposal_losses = self.detector.rpn(images, features, targets)
        detections, detector_losses = self.detector.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.detector.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        output_dict = detections[0]

        original_person_ls = []
        original_detect_res, possible_hidden_res = [], []
        if len(output_dict["boxes"]) > 0:
            out_boxes = output_dict["boxes"].cpu()
            out_scores = output_dict["scores"].cpu()
            out_labels = output_dict["labels"].cpu()

            for idx in range(0, min(out_boxes.shape[0], 300)):

                score = out_scores[idx].detach().numpy()
                bbox = out_boxes[idx].detach().numpy()
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cls_pred = out_labels[idx].detach().numpy()

                if idx <= self.box_detections_per_img:
                    if score >= self.thres:
                        if cls_pred == 1:
                            high_overlap = False
                            for single_boxes in original_person_ls:
                                if single_boxes[-1] == 1:
                                    tmp_overlap = cal_overlap_only([x1, y1, x2, y2], single_boxes[:4], x1y1x2y2=True)
                                    if tmp_overlap >=0.8:
                                        high_overlap = True
                                        break
                            if not high_overlap:
                                original_detect_res.append([x1, y1, x2, y2, score, cls_pred])
                                original_person_ls.append([x1, y1, x2, y2, score, cls_pred])
                        else:
                            original_detect_res.append([x1, y1, x2, y2, score, cls_pred])
                    if (cls_pred==interest_cls) and (score>=interest_score_thresh):
                        possible_hidden_res.append([x1, y1, x2, y2, score, cls_pred])

        original_detect_res, possible_hidden_res = np.array(original_detect_res), np.array(possible_hidden_res)
        original_person_ls = np.array(original_person_ls)
        hidden_xyxy = []
        for single_possible in possible_hidden_res:
            tmp_overlap = False
            for single_original_person in original_person_ls:
                tmp_overlap = cal_overlap_only(single_original_person[:4], single_possible[:4], x1y1x2y2=True)
                if tmp_overlap >= 0.4:
                    tmp_overlap = True
                    break
            if not tmp_overlap:
                hidden_xyxy.append( single_possible )
        hidden_xyxy = np.array(hidden_xyxy)

        if len(hidden_xyxy) >1:
            hidden_xyxy = divid_person_group(hidden_xyxy)
            hidden_box = np.ones((len(hidden_xyxy), 6))
            hidden_box[:, :4] = hidden_xyxy
            hidden_xyxy = hidden_box


        return original_detect_res, hidden_xyxy


    def draw_possible_area_with_boxes(self, img_cv, bboxes):
        img_original_cv = copy.deepcopy(img_cv)
        if len(bboxes)>0:
            for single_res in bboxes:
                x1, y1, x2, y2, score, cls_pred = single_res
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img_original_cv, (x1, y1), (x2, y2), (255,255,255), thickness=2)

        return img_original_cv

    def get_possible_area_for_fold(self, img_path, save_path):
        os.makedirs(save_path, exist_ok=True)
        for img_name in os.listdir(img_path):
            img_cv = cv2.imread(os.path.join(img_path, img_name), 1)
            original_res, possible_res = self.detect_for_hidden_person(img_cv)
            img_possible = self.draw_possible_area_with_boxes(img_cv, possible_res)
            cv2.imwrite(os.path.join(save_path, img_name), img_possible)
        pass



