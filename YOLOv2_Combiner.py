import os
import cv2
import torch
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

import json
import argparse
import datetime
import shutil
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from ObjectDetector.fjn_yolov2 import *

def freeze_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def reformat_to_label(max_lab, batch_size, label, original_label):
    label_ls = []
    for b_id in range(batch_size):
        tmp_label = torch.zeros((max_lab, 5))
        tmp_label[:, 0] = -1
        tmp_label_id = 0
        if label is not None:
            for label_id in range(len(label[b_id])):
                if label_id < max_lab:
                    tmp_label[label_id][0] = label[b_id][label_id][0]
                    tmp_label[label_id][1] = label[b_id][label_id][1]
                    tmp_label[label_id][2] = label[b_id][label_id][2]
                    tmp_label[label_id][3] = label[b_id][label_id][3]
                    tmp_label[label_id][4] = label[b_id][label_id][4]
                    tmp_label_id += 1
        if original_label is not None:
            for original_label_id in range(len(original_label[b_id])):
                if original_label_id + tmp_label_id < max_lab:
                    tmp_label[tmp_label_id + original_label_id][0] = original_label[b_id][original_label_id][0]
                    tmp_label[tmp_label_id + original_label_id][1] = original_label[b_id][original_label_id][1]
                    tmp_label[tmp_label_id + original_label_id][2] = original_label[b_id][original_label_id][2]
                    tmp_label[tmp_label_id + original_label_id][3] = original_label[b_id][original_label_id][3]
                    tmp_label[tmp_label_id + original_label_id][4] = original_label[b_id][original_label_id][4]
                else:
                    break
        label_ls.append(tmp_label)
    if len(label_ls) == 1:
        return label_ls[0].unsqueeze(0)
    else:
        res = label_ls[0].unsqueeze(0)
        for j in range(1, len(label_ls)):
            res = torch.cat([res, label_ls[j].unsqueeze(0)], dim=0)
        return res

class Yolov2Dataset(Dataset):
    def __init__(self, benign_root, adversarial_root, benign_label_root, img_size, max_label_num):
        self.benign_root = benign_root
        self.benign_label_root = benign_label_root
        self.adversarial_root = adversarial_root
        self.img_size = img_size
        self.img_benign_ls = os.listdir(self.benign_root)
        self.img_adversarial_ls = os.listdir(self.adversarial_root)
        assert len(self.img_benign_ls) == len(
            self.img_adversarial_ls), 'the number of benign images should equal to adversarial images'
        self.img_ls_len = len(self.img_benign_ls)
        self.max_original_label_num = max_label_num

    def __len__(self):
        return self.img_ls_len

    def __getitem__(self, idx):
        img_name = self.img_benign_ls[idx % self.img_ls_len]
        img_benign_path = os.path.join(self.benign_root, img_name)
        img_benign_input_original = cv2.imread(img_benign_path, 1)
        img_benign_input = cv2.resize(img_benign_input_original, (self.img_size, self.img_size))
        img_benign_input = cv2.cvtColor(img_benign_input, cv2.COLOR_BGR2RGB)
        img_benign_input_tensor = torch.from_numpy(np.transpose(img_benign_input, [2, 0, 1])) / 255.
        label_benign_txt_name = img_name.split('.')[0] + '.txt'
        label_benign_tensor = self.read_benign_label(os.path.join(self.benign_label_root, label_benign_txt_name))
        img_adversarial_path = os.path.join(self.adversarial_root, img_name)
        img_adversarial_input_original = cv2.imread(img_adversarial_path, 1)
        img_adversarial_input = cv2.resize(img_adversarial_input_original, (self.img_size, self.img_size))
        img_adversarial_input = cv2.cvtColor(img_adversarial_input, cv2.COLOR_BGR2RGB)
        img_adversarial_input_tensor = torch.from_numpy(np.transpose(img_adversarial_input, [2, 0, 1])) / 255.
        return img_benign_input_tensor, label_benign_tensor, img_adversarial_input_tensor

    def reformat_patch_area(self, place_np):
        max_place_num = self.max_original_label_num
        place_np_res = np.zeros((max_place_num, 4))
        if len(place_np.shape) == 0 or len(place_np) == 0:
            return place_np_res
        else:
            if len(place_np.shape) == 1:
                place_np = np.expand_dims(place_np, axis=0)
            for i in range(place_np.shape[0]):
                if i >= max_place_num:
                    break
                place_np_res[i] = place_np[i]
            return place_np_res

    def read_benign_label(self, label_path):
        label = np.loadtxt(label_path)
        if len(label) == 0:
            label = np.array([[-1, 0, 0, 0, 0]])
        elif len(label.shape) == 1:
            label = np.expand_dims(label, axis=0)
        label_tensor = torch.zeros((self.max_original_label_num, 5))
        label_tensor[:, 0] = -1
        for i in range(label.shape[0]):
            if i < self.max_original_label_num:
                label_tensor[i][0] = label[i][0]
                label_tensor[i][1] = label[i][1]
                label_tensor[i][2] = label[i][2]
                label_tensor[i][3] = label[i][3]
                label_tensor[i][4] = label[i][4]
        return label_tensor

def add_defensive_patch_into_img_tensor(cfg, detector, img_tensor, df_patch_tensor):
    add_df_patch_clsxywh = []
    for bi in range(img_tensor.shape[0]):
        possible_person = FindHiddenPerson(detector, img_tensor[bi:bi + 1, ::], person_conf=cfg.person_conf, overlap_thresh=cfg.overlap_thresh)
        tmp_df_patch_clsxywh = []
        if len(possible_person) > 0:
            for single_possible in possible_person:
                possible_x, possible_y = int(single_possible[0] * cfg.img_size), int(single_possible[1] * cfg.img_size)
                possible_w, possible_h = int(single_possible[2] * cfg.img_size), int(single_possible[3] * cfg.img_size)
                location_y, location_x = cfg.defensive_patch_location[0], cfg.defensive_patch_location[1]
                if location_y == 'u':
                    start_y = possible_y - 30 - df_patch_tensor.shape[-1] // 2
                elif location_y == 'c':
                    start_y = possible_y - df_patch_tensor.shape[-1] // 2
                elif location_y == 'b':
                    start_y = possible_y + 30 - df_patch_tensor.shape[-1] // 2
                else:
                    raise Exception('add_defensive_patch_into_img_tensor: no such defensive_patch_location: y ', cfg.defensive_patch_location)
                if location_x == 'l':
                    start_x = possible_x - 30 - df_patch_tensor.shape[-1] // 2
                elif location_x == 'c':
                    start_x = possible_x - df_patch_tensor.shape[-1] // 2
                elif location_x == 'r':
                    start_x = possible_x + 30 - df_patch_tensor.shape[-1] // 2
                else:
                    raise Exception('add_defensive_patch_into_img_tensor: no such defensive_patch_location: x ', cfg.defensive_patch_location)
                end_x = start_x + df_patch_tensor.shape[-2]
                end_y = start_y + df_patch_tensor.shape[-1]
                if cfg.train:
                    shift_x = random.randint(-10, 10)
                    shift_y = random.randint(-10, 10)
                    start_y, end_y = start_y + shift_y, end_y + shift_y
                    start_x, end_x = start_x + shift_x, end_x + shift_x
                if end_y > cfg.img_size:
                    end_y = cfg.img_size - cfg.margin_size
                    start_y = end_y - df_patch_tensor.shape[-1]
                if end_x > cfg.img_size:
                    end_x = cfg.img_size - cfg.margin_size
                    start_x = end_x - df_patch_tensor.shape[-2]
                if start_y < 0:
                    start_y = cfg.margin_size
                    end_y = df_patch_tensor.shape[-1] + cfg.margin_size
                if start_x < 0:
                    start_x = cfg.margin_size
                    end_x = df_patch_tensor.shape[-2] + cfg.margin_size
                img_tensor[bi, :, start_y:end_y, start_x:end_x] = df_patch_tensor
                tmp_df_patch_clsxywh.append(
                    [cfg.canary_cls_id, (start_x + end_x) / 2 / cfg.img_size, (start_y + end_y) / 2 / cfg.img_size, df_patch_tensor.shape[-2] / cfg.img_size, df_patch_tensor.shape[-1] / cfg.img_size])
        add_df_patch_clsxywh.append(tmp_df_patch_clsxywh)

    return img_tensor, add_df_patch_clsxywh

def add_defensive_patch_into_img_cv(cfg, possible_person_ls, img_cv_in, df_patch_cv, has_area=False):
    img_cv = img_cv_in.copy()
    add_df_patch_clsxywh = []
    for single_person_xywh in possible_person_ls:
        if np.count_nonzero(single_person_xywh) == 0:
            continue
        possible_x, possible_y = int(single_person_xywh[0] * cfg.img_size), int(single_person_xywh[1] * cfg.img_size)
        possible_w, possible_h = int(single_person_xywh[2] * cfg.img_size), int(single_person_xywh[3] * cfg.img_size)
        location_y, location_x = cfg.defensive_patch_location[0], cfg.defensive_patch_location[1]
        if location_y == 'u':
            start_y = possible_y - 30 - df_patch_cv.shape[0] // 2
        elif location_y == 'c':
            start_y = possible_y - df_patch_cv.shape[1] // 2
        elif location_y == 'b':
            start_y = possible_y + 30 - df_patch_cv.shape[0] // 2
        else:
            raise Exception('add_defensive_patch_into_img_tensor: no such defensive_patch_location: y ', cfg.defensive_patch_location)
        if location_x == 'l':
            start_x = possible_x - 30 - df_patch_cv.shape[0] // 2
        elif location_x == 'c':
            start_x = possible_x - df_patch_cv.shape[1] // 2
        elif location_x == 'r':
            start_x = possible_x + 30 - df_patch_cv.shape[0] // 2
        else:
            raise Exception('add_defensive_patch_into_img_tensor: no such defensive_patch_location: x ', cfg.defensive_patch_location)
        end_x = start_x + df_patch_cv.shape[1]
        end_y = start_y + df_patch_cv.shape[0]
        if end_y > cfg.img_size:
            end_y = cfg.img_size - cfg.margin_size
            start_y = end_y - df_patch_cv.shape[1]
        if end_x > cfg.img_size:
            end_x = cfg.img_size - cfg.margin_size
            start_x = end_x - df_patch_cv.shape[0]
        if start_y < 0:
            start_y = cfg.margin_size
            end_y = df_patch_cv.shape[1] + cfg.margin_size
        if start_x < 0:
            start_x = cfg.margin_size
            end_x = df_patch_cv.shape[0] + cfg.margin_size
        img_cv[start_y:end_y, start_x:end_x, :] = df_patch_cv
        add_df_patch_clsxywh.append([(start_x + end_x) / 2 / cfg.img_size, (start_y + end_y) / 2 / cfg.img_size, df_patch_cv.shape[1] / cfg.img_size, df_patch_cv.shape[0] / cfg.img_size])
    if has_area:
        return img_cv, add_df_patch_clsxywh
    else:
        return img_cv

def box_in_another_box(single_box_original_person, single_box_all_person):
    bool_x_in_left = (single_box_original_person[0] - single_box_original_person[2] / 2) < single_box_all_person[0]
    if not bool_x_in_left:
        return False

    bool_x_in_right = single_box_all_person[0] < (single_box_original_person[0] + single_box_original_person[2] / 2)
    if not bool_x_in_right:
        return False

    bool_y_in_top = (single_box_original_person[1] - single_box_original_person[3] / 2) < single_box_all_person[1]
    if not bool_y_in_top:
        return False

    bool_y_in_bottom = single_box_all_person[1] < (single_box_original_person[1] + single_box_original_person[3] / 2)
    if not bool_y_in_bottom:
        return False

    return True

def FindHiddenPerson(detector, img, person_conf=0.05, overlap_thresh=0.4):
    box_all_person, box_original_obj = detector.detect_person_and_original(img, person_conf=person_conf, faster=True)
    box_original_person = box_original_obj[np.where(box_original_obj[:, -1] == 0)]
    box_all_person_overlap = []
    if len(box_original_person) > 0:
        for single_box_all_person in box_all_person:
            boo_is_overlap = False
            for single_box_original_person in box_original_person:
                if box_in_another_box(single_box_original_person, single_box_all_person):
                    boo_is_overlap = True
                    break
                if cal_overlap(single_box_all_person[:4], single_box_original_person[:4], x1y1x2y2=False) >= overlap_thresh:
                    boo_is_overlap = True
                    break
            if not boo_is_overlap:
                box_all_person_overlap.append(single_box_all_person)
    else:
        box_all_person_overlap = box_all_person
    new_box_all_person_overlap = []
    for single_box_person_overlap in box_all_person_overlap:
        bool_tmp_is_defense_overlap = False
        for single_original_person_box in box_original_person:
            tmp_overlap_area = cal_overlap_only([min(max(single_box_person_overlap[0], 0.131), 0.868), min(max(single_box_person_overlap[1], 0.131), 0.868), 0.098, 0.098], single_original_person_box, x1y1x2y2=False)
            if tmp_overlap_area / (0.098 * 0.098) > 0.3:
                bool_tmp_is_defense_overlap = True

        if not bool_tmp_is_defense_overlap:
            new_box_all_person_overlap.append(single_box_person_overlap)
            pass
        pass
    if len(new_box_all_person_overlap) > 0:
        new_all_person_boxes = np.array(new_box_all_person_overlap)
        new_all_person_xyxy_boxes = xywh2xyxy_np(new_all_person_boxes[:, :4])
        new_all_person_xyxy_boxes = np.minimum(np.maximum(new_all_person_xyxy_boxes, 0), 1.0)
        person_groups_xyxy = divid_person_group(new_all_person_xyxy_boxes)
        hidden_xywh = xyxy2xywh_np(person_groups_xyxy)
        hidden_box = []
        for hidden_xywh_item in hidden_xywh:
            if hidden_xywh_item[2]>0.05 and hidden_xywh_item[3]>0.05:
                hidden_box_single = np.zeros((7))
                hidden_box_single[:4] = hidden_xywh_item
                hidden_box.append(hidden_box_single)
        hidden_box = np.array( hidden_box )
        hidden_xywh = hidden_box
    else:
        hidden_xywh = np.array([])

    return hidden_xywh

class Woodpecker():
    def __init__(self, cfg, detector):
        self.cfg = cfg
        self.detector = detector
        self.data_loader = None
        self.wd_tensor = None
        self.wd_save_path = ''
        self.wd_cv = None
        pass

    def train(self):
        region_loss = self.detector.model.loss
        optimizer = optim.Adam([self.wd_tensor], lr=self.cfg.learing_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epoch)
        t = tqdm(total=self.cfg.epoch, ascii=True)
        for epoch in range(1, cfg.epoch + 1):
            t.set_description((f'wd_{self.cfg.attack_name}_{self.cfg.detect_name}_{cfg.weight} epoch: {epoch}/{self.cfg.epoch}'))
            for i_batch, (benign_input_tensor, label_benign_tensor, adv_input_tensor) in enumerate(self.data_loader):
                benign_input_tensor = benign_input_tensor.requires_grad_(False).cuda()
                adv_input_tensor = adv_input_tensor.requires_grad_(False).cuda()
                benign_input_wd_tensor, _ = add_defensive_patch_into_img_tensor(cfg, detector, img_tensor=benign_input_tensor, df_patch_tensor=self.wd_tensor)
                adv_input_wd_tensor, _ = add_defensive_patch_into_img_tensor(cfg, detector, img_tensor=adv_input_tensor, df_patch_tensor=self.wd_tensor)
                benign_label_clsxywh = reformat_to_label(max_lab=self.cfg.max_label, batch_size=self.cfg.batch_size, label=None, original_label=label_benign_tensor)
                adv_label_clsxywh = reformat_to_label(max_lab=self.cfg.max_label, batch_size=self.cfg.batch_size, label=None, original_label=label_benign_tensor)
                benign_label_clsxywh = torch.reshape(benign_label_clsxywh, (benign_label_clsxywh.shape[0], -1))
                adv_label_clsxywh = torch.reshape(adv_label_clsxywh, (adv_label_clsxywh.shape[0], -1))
                benign_outputs = self.detector.model(benign_input_wd_tensor)
                benign_box_l, benign_obj_l, benign_cls_l = region_loss(benign_outputs, benign_label_clsxywh, self.cfg.max_label)
                adv_outputs = self.detector.model(adv_input_wd_tensor)
                adv_hidden_box_l, adv_hidden_obj_l, adv_hidden_cls_l = region_loss(adv_outputs, adv_label_clsxywh, self.cfg.max_label)
                benign_loss = benign_box_l + benign_obj_l + benign_cls_l
                adv_loss = adv_hidden_box_l + adv_hidden_obj_l + adv_hidden_cls_l
                loss = self.cfg.weight * benign_loss + adv_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if not cfg.defensive_patch_clip:
                    self.wd_tensor.requires_grad_(True)
                    self.wd_tensor.data.clamp_(0.0001, 0.9999)
                t.set_postfix({
                    'benign_loss': '{0:1.5f}'.format(benign_loss / len(self.data_loader)),
                    'adv_loss': '{0:1.5f}'.format(adv_loss / len(self.data_loader)),
                })
                if epoch % self.cfg.epoch_save == 0:
                    self.save_wd(epoch)
            t.update(1)
            scheduler.step()
        t.close()
        pass

    def eval_single(self, img_cv):
        possible_person_ls = FindHiddenPerson(self.detector, img_cv, person_conf=self.cfg.person_conf, overlap_thresh=self.cfg.overlap_thresh)
        img_sized = cv2.resize(img_cv, (self.cfg.img_size, self.cfg.img_size))
        if len(possible_person_ls) == 0 or (np.count_nonzero(possible_person_ls) == 0):
            return False
        else:
            img_original_result = self.detector.detect_single(img_sized)
            detected_person_in_original_ls = img_original_result[np.where(img_original_result[:, -1] == 0)]
            img_sized_with_wd = add_defensive_patch_into_img_cv(self.cfg, possible_person_ls, img_sized, self.wd_cv, has_area=False)
            img_with_wd_result = self.detector.detect_single(img_sized_with_wd)
            detected_person_in_wd_ls = img_with_wd_result[np.where(img_with_wd_result[:, -1] == 0)]
            has_new_person = False
            if len(detected_person_in_original_ls) == 0 and len(detected_person_in_wd_ls) == 0:
                has_new_person = False
            elif len(detected_person_in_original_ls) == 0 and len(detected_person_in_wd_ls) > 0:
                has_new_person = True
            else:
                for new_single_person in detected_person_in_wd_ls:
                    temp_ls = []
                    for old_single_person in detected_person_in_original_ls:
                        chongdie = self.cal_chongdie(new_single_person[:4], old_single_person[:4], x1y1x2y2=False)
                        temp_ls.append(chongdie)
                    if max(temp_ls) < 0.4:
                        has_new_person = True
                        break
            return has_new_person

    def save_wd(self, epoch):
        if not cfg.defensive_patch_clip:
            wd_img_RGB = transforms.ToPILImage('RGB')(self.wd_tensor)
            wd_img_RGB = np.array(wd_img_RGB).astype(np.uint8)
            wd_img_BGR = cv2.cvtColor(wd_img_RGB, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(f'{self.wd_save_path}', f'wd_{str(epoch).zfill(3)}.png'), wd_img_BGR)
            del wd_img_BGR
        else:
            wd_cv_RGB = self.wd_tensor.data.numpy()
            wd_cv_BGR = np.flip(wd_cv_RGB, axis=2)
            np.save(os.path.join(f'{self.wd_save_path}', f'wd_{str(epoch).zfill(3)}.npy'), wd_cv_BGR)
        pass

    def init_woodpecker(self):
        wd_img = np.random.randint(0, 255, (3, self.cfg.wd_size, self.cfg.wd_size)) / 255.
        self.wd_cv = (wd_img.transpose(1, 2, 0) * 255).astype(np.uint8)
        wd_tensor = torch.from_numpy(wd_img)
        self.wd_tensor = wd_tensor.requires_grad_(True)
        fold_name = f'wd_{self.cfg.defensive_patch_location}_{self.cfg.attack_name}_{self.cfg.detect_name}_{self.cfg.weight}_{self.cfg.batch_size}'
        wd_fold = f'exp_{self.cfg.data_name}_{self.cfg.wd_size}_{self.cfg.epoch}'
        self.wd_save_path = os.path.join('./FJNTraining', fold_name, wd_fold)
        os.makedirs(self.wd_save_path, exist_ok=True)
        save_cfg_content = {}
        for arg in vars(self.cfg):
            save_cfg_content.update({arg: getattr(cfg, arg)})
        with open(os.path.join(self.wd_save_path, 'cfg.json' ), "w") as f:
            json.dump(save_cfg_content, f)
        self.save_wd(0)
        del wd_img

        pass

    def load_eval_wd(self, wd_path):
        if not os.path.exists(wd_path):
            print('Load woodpecker',  wd_path, 'not exists!' )
            exit()
        if not self.cfg.defensive_patch_clip:
            wd_cv = cv2.imread(wd_path, 1)
            if wd_cv.shape[0] != self.cfg.wd_size or wd_cv.shape[1] != self.cfg.wd_size:
                print('Load Woodpecker', wd_path, ' fail: img_size ', wd_cv.shape, 'not match set_size', self.cfg.wd_size)
                exit()
            self.wd_cv = wd_cv
            print(f'Load Woodpecker for eval: {wd_path}')
        else:
            wd_cv = np.load(wd_path)
            wd_cv = np.transpose(wd_cv, (1,2,0))
            self.wd_cv = wd_cv * 255.
            print(f'Load Woodpecker for eval: {wd_path}')
        pass

    def cal_chongdie(self, box1, box2, x1y1x2y2=False):
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
        if cw <= 0 or ch <= 0:
            carea = 0
        else:
            carea = cw * ch
        return carea / area1
    
    def init_dataloader(self):
        train_dataset = Yolov2Dataset(self.cfg.benign_root, self.cfg.adversarial_root, self.cfg.benign_label_root, self.cfg.img_size, self.cfg.max_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_works, drop_last=True)
        self.data_loader = train_loader
        pass

class Canary:
    def __init__(self, cfg, detector):
        self.cfg = cfg
        self.detector = detector
        self.img_size = [self.cfg.img_size, self.cfg.img_size]
        self.person_conf = self.cfg.person_conf
        self.overlap_thresh = self.cfg.overlap_thresh
        self.data_loader = None
        self.canary_tensor = None
        self.canary_save_path = ''
        self.canary_cv = None
        self.canary_cls_id = -1
        pass

    def train(self):
        region_loss = self.detector.model.loss
        optimizer = optim.Adam([self.canary_tensor], lr=self.cfg.learing_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epoch)
        t = tqdm(total=self.cfg.epoch, ascii=True)
        for epoch in range(1, self.cfg.epoch + 1):
            t.set_description((f'canary_{self.cfg.attack_name}_{self.cfg.detect_name}_{self.cfg.weight} epoch: {epoch}/{self.cfg.epoch}'))
            for i_batch, (benign_input_tensor, label_benign_tensor, adv_input_tensor) in enumerate(self.data_loader):
                benign_input_tensor = benign_input_tensor.requires_grad_(False).cuda()
                adv_input_tensor = adv_input_tensor.requires_grad_(False).cuda()
                adv_input_canary_tensor, adv_canary_clsxywh = add_defensive_patch_into_img_tensor(cfg, detector, img_tensor=adv_input_tensor, df_patch_tensor=self.canary_tensor)
                benign_input_canary_tensor, benign_canary_clsxywh = add_defensive_patch_into_img_tensor(cfg, detector, img_tensor=benign_input_tensor, df_patch_tensor=self.canary_tensor)
                benign_label_clsxywh = reformat_to_label(max_lab=self.cfg.max_label, batch_size=self.cfg.batch_size, label=benign_canary_clsxywh, original_label=label_benign_tensor)
                adv_label_clsxywh = reformat_to_label(max_lab=self.cfg.max_label, batch_size=self.cfg.batch_size, label=adv_canary_clsxywh, original_label=None)
                benign_label_clsxywh = torch.reshape(benign_label_clsxywh, (benign_label_clsxywh.shape[0], -1))
                adv_label_clsxywh = torch.reshape(adv_label_clsxywh, (adv_label_clsxywh.shape[0], -1))
                benign_outputs = self.detector.model(benign_input_canary_tensor)
                benign_box_l, benign_obj_l, benign_cls_l = region_loss(benign_outputs, benign_label_clsxywh, self.cfg.max_label)
                adv_outputs = self.detector.model(adv_input_canary_tensor)
                adv_hidden_box_l, adv_hidden_obj_l, adv_hidden_cls_l = region_loss(adv_outputs, adv_label_clsxywh, self.cfg.max_label)
                benign_loss = benign_obj_l + benign_cls_l + benign_box_l
                adv_loss = adv_hidden_obj_l + adv_hidden_cls_l+ adv_hidden_box_l
                loss = self.cfg.weight * benign_loss - adv_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not cfg.defensive_patch_clip:
                    self.canary_tensor.requires_grad_(True)
                    self.canary_tensor.data.clamp_(0.0001, 0.9999)
                t.set_postfix({
                    'benign_loss': '{0:1.5f}'.format(benign_loss / len(self.data_loader)),
                    'adv_loss': '{0:1.5f}'.format(adv_loss / len(self.data_loader)),
                })
            if epoch % self.cfg.epoch_save == 0:
                self.save_canary(epoch)
            t.update(1)
            scheduler.step()
        t.close()
        pass

    def save_canary(self, epoch):
        if not cfg.defensive_patch_clip:
            canary_cv_RGB = transforms.ToPILImage('RGB')(self.canary_tensor)
            canary_cv_RGB = np.array(canary_cv_RGB).astype(np.uint8)
            canary_cv_BGR = cv2.cvtColor(canary_cv_RGB, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(f'{self.canary_save_path}', f'canary_{str(epoch).zfill(3)}.png'), canary_cv_BGR)
            del canary_cv_BGR
        else:
            canary_cv_RGB = self.canary_tensor.data.numpy()
            canary_cv_BGR = np.flip(canary_cv_RGB, axis=2)
            np.save(os.path.join( f'{self.canary_save_path}', f'canary_{str(epoch).zfill(3)}.npy' ), canary_cv_BGR)
            pass

        pass

    def init_canary(self):
        if self.cfg.canary_init and os.path.exists(os.path.join(self.cfg.canary_init_path, f'{self.cfg.canary_cls_id}.jpg')):
            canary_cv = cv2.imread(os.path.join(self.cfg.canary_init_path, f'{self.cfg.canary_cls_id}.jpg'), 1)
            canary_cv = cv2.resize(canary_cv, (self.cfg.ca_size, self.cfg.ca_size))
            self.canary_cv = canary_cv
            canary_cv = np.transpose(canary_cv, (2, 0, 1)) / 255.
        else:
            canary_cv = np.ones((3, self.cfg.ca_size, self.cfg.ca_size)) * 0.5
            self.canary_cv = (np.ones((self.cfg.ca_size, self.cfg.ca_size, 3)) * 127).astype(np.uint8)
        canary_tensor = torch.from_numpy(canary_cv)
        self.canary_tensor = canary_tensor.requires_grad_(True)
        self.canary_cls_id = self.cfg.canary_cls_id
        fold_name = f'canary_{self.cfg.defensive_patch_location}_{self.cfg.attack_name}_{self.cfg.detect_name}_{self.cfg.weight}_{self.cfg.batch_size}'
        canary_fold = f'exp_{self.cfg.data_name}_{self.cfg.canary_cls_id}_{self.cfg.ca_size}_{self.cfg.epoch}'
        self.canary_save_path = os.path.join('./FJNTraining', fold_name, canary_fold)
        os.makedirs(self.canary_save_path, exist_ok=True)
        del canary_cv, fold_name, canary_fold
        save_cfg_content = {}
        for arg in vars(self.cfg):
            save_cfg_content.update({arg: getattr(cfg, arg)})
        with open(os.path.join(self.canary_save_path, 'cfg.json' ), "w") as f:
            json.dump(save_cfg_content, f)
        self.save_canary(0)
        pass

    def eval_single(self, img_cv):
        possible_person_ls = FindHiddenPerson(self.detector, img_cv, person_conf=self.cfg.person_conf, overlap_thresh=self.cfg.overlap_thresh)
        img_sized = cv2.resize(img_cv, (self.cfg.img_size, self.cfg.img_size))
        if len(possible_person_ls) == 0 or (np.count_nonzero(possible_person_ls) == 0):
            return False
        else:
            img_original_result = self.detector.detect_single(img_sized)
            original_canary_result = img_original_result[np.where(img_original_result[:, -1] == self.canary_cls_id)]
            img_sized_with_canary = copy.deepcopy(img_sized)
            img_sized_with_canary, canary_area = add_defensive_patch_into_img_cv(self.cfg, possible_person_ls, img_sized_with_canary, self.canary_cv, has_area=True)
            img_with_canary_result = self.detector.detect_single(img_sized_with_canary)
            canary_result = img_with_canary_result[np.where(img_with_canary_result[:, -1] == self.canary_cls_id)]
            canary_original_num = len(original_canary_result)
            canary_put_num = len(canary_area)
            canary_detected_num = len(canary_result)
            if canary_original_num + canary_put_num == canary_detected_num:
                is_attack = False
            else:
                is_attack = True
        return is_attack

    def load_eval_canary(self, canary_path, canary_cls_id):
        if not os.path.exists(canary_path):
            print('Load Canary',  canary_path, 'not exists!' )
            exit()
        if not self.cfg.defensive_patch_clip:
            canary_cv = cv2.imread(canary_path, 1)
            if canary_cv.shape[0] != self.cfg.ca_size or canary_cv.shape[1] != self.cfg.ca_size:
                raise Exception('Load Canary', canary_path, ' fail: img_size ', canary_cv.shape, 'not match set_size', self.cfg.ca_size)
            self.canary_cv = canary_cv
            self.canary_cls_id = canary_cls_id
            print(f'Load canary({canary_cls_id}) for eval: {canary_path}')
        else:
            canary_cv = np.load(canary_path)
            canary_cv = np.transpose(canary_cv, (1,2,0))
            self.canary_cv = canary_cv * 255.
            self.canary_cls_id = canary_cls_id
            print(f'Load canary({canary_cls_id}) for eval: {canary_path}')
            pass
        pass
    
    def init_dataloader(self):
        train_dataset = Yolov2Dataset(self.cfg.benign_root, self.cfg.adversarial_root, self.cfg.benign_label_root, self.cfg.img_size, self.cfg.max_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_works, drop_last=True)
        self.data_loader = train_loader
        pass


def get_args():
    Gparser = argparse.ArgumentParser(description='Train normal defensive patch')
    Gparser.add_argument('--detect_name', default='yolov2', type=str, help='')
    Gparser.add_argument('--attack_name', default='fool', type=str, help='', choices=['fool', 'advtext', 'upc', 'natural'])
    Gparser.add_argument('--weight', default=1.0, type=float, help='weight of benign loss')
    Gparser.add_argument('--epoch', default=50, type=int, help='epoch for train')
    Gparser.add_argument('--learing_rate', default=0.05, type=float, help='batch_size for training')
    Gparser.add_argument('--epoch_save', default=1, type=int, help='epoch for save model and defensive patch')
    Gparser.add_argument('--img_size', default=608, type=int)
    Gparser.add_argument('--data_name', default='VOC07_120_New', type=str, help='')
    Gparser.add_argument('--benign_root', default="Data/traineval/VOC07_YOLOv2/train_120_New0603/benign", type=str)
    Gparser.add_argument('--benign_label_root', default="Data/traineval/VOC07_YOLOv2/train_120_New0603/benign_label", type=str)
    Gparser.add_argument('--adversarial_root', default="Data/traineval/VOC07_YOLOv2/train_120_New0603/adversarial", type=str)
    Gparser.add_argument('--batch_size', default=10, type=int, help='batch_size for training')
    Gparser.add_argument('--num_works', default=5, type=int, help='num_works')
    Gparser.add_argument('--max_label', default=20, type=int, help='max_label')
    Gparser.add_argument('--margin_size', default=20, type=int, help='margin size')
    Gparser.add_argument('--person_conf', default=0.05, type=float, help='person_conf')
    Gparser.add_argument('--overlap_thresh', default=0.4, type=float, help='overlap_thresh')
    Gparser.add_argument('--defensive_patch_size', default=60, type=int, help='defensive patch size')
    Gparser.add_argument('--defensive_patch_clip', action='store_true', default=False, help='options :True or False')
    Gparser.add_argument('--defensive_patch_location', default='cc', type=str, help='defensive patch location', choices=['uc', 'cl', 'cr', 'bc', 'cc'])
    Gparser.add_argument('--canary_init_path', default='InitImages/', type=str, help='canary init image')
    Gparser.add_argument('--canary_init', action='store_true', default=True, help='options :True or False')
    Gparser.add_argument('--canary_cls_id', default=22, type=int, help='canary label')
    Gparser.add_argument('--ca_size', default=60, type=int, help='defensive patch size')
    Gparser.add_argument('--wd_size', default=60, type=int, help='defensive patch size')
    Gparser.add_argument('--shuffle', action='store_true', default=True, help='options :True or False')
    Gparser.add_argument('--seed', default=301, type=int, help='choose seed')
    Gparser.add_argument('--df_mode', default='C', type=str, help='select df_patch', choices=['C', 'W', 'A'])
    Gparser.add_argument('--gpu_id', default=0, type=int, help='choose gpu')
    Gparser.add_argument('--train', action='store_true', help='train')
    Gparser.add_argument('--test', action='store_true', help='test')
    Gparser.add_argument('--input_img', default='', type=str, help='the patch of input img')
    Gparser.add_argument('--best_canary_path', default='./trained_dfpatches/YOLOv2/canary.png', type=str, help='the patch of best_canary')
    Gparser.add_argument('--best_wd_path', default='./trained_dfpatches/YOLOv2/wd.png', type=str, help='the patch of best_wd')
    return Gparser.parse_known_args()[0]


'''
python YOLOv2_Combiner.py --train --df_mode C --defensive_patch_location cc --canary_cls_id 22 --ca_size 60 --person_conf 0.05 --weight 2.0
python YOLOv2_Combiner.py --train --df_mode W --defensive_patch_location cc --wd_size 60 --person_conf 0.05 --weight 1.0

python YOLOv2_Combiner.py --test --df_mode C --defensive_patch_location cc --canary_cls_id 22 --ca_size 60 --person_conf 0.05 --best_canary_path ./trained_dfpatches/YOLOv2/canary.png --input_img XXX
python YOLOv2_Combiner.py --test --df_mode W --defensive_patch_location cc --wd_size 60 --person_conf 0.05 --best_wd_path ./trained_dfpatches/YOLOv2/wd.png --input_img XXX
python YOLOv2_Combiner.py --test --df_mode A --defensive_patch_location cc --canary_cls_id 22 --ca_size 60 --wd_size 60 --person_conf 0.05 --best_canary_path ./trained_dfpatches/YOLOv2/canary.png --best_wd_path ./trained_dfpatches/YOLOv2/wd.png --input_img XXX

'''

if __name__ == '__main__':
    cfg = get_args()
    freeze_seed(cfg.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.gpu_id}'
    detector = FJN_YOLOV2()
    
    if cfg.train:
        if cfg.df_mode == 'C':
            canary = Canary(cfg, detector)
            canary.init_dataloader()
            canary.init_canary()
            canary.train()
            pass
        if cfg.df_mode == 'W':
            woodpecker = Woodpecker(cfg, detector)
            woodpecker.init_dataloader()
            woodpecker.init_woodpecker()
            woodpecker.train()
            pass
        pass
    if cfg.test:
        if cfg.df_mode == 'C' or cfg.df_mode == 'A' :
            canary = Canary(cfg, detector)
            canary.load_eval_canary(canary_path=cfg.best_canary_path, canary_cls_id=cfg.canary_cls_id)
        if cfg.df_mode == 'W' or cfg.df_mode == 'A' :
            woodpecker = Woodpecker(cfg, detector)
            woodpecker.load_eval_wd(wd_path=cfg.best_wd_path)
        img_cv = cv2.imread(cfg.input_img, 1)
        
        if cfg.df_mode == 'C':
            is_attack = canary.eval_single(img_cv)
        elif cfg.df_mode == 'W':
            is_attack = woodpecker.eval_single(img_cv)
        elif cfg.df_mode == 'A':
            is_attack = canary.eval_single(img_cv)
            if not is_attack:
                is_attack = woodpecker.eval_single(img_cv)
        if is_attack:
            print('detect adversarial attack!')
        else:
            print('not detect adversarial attack!')

