import copy
import os
import cv2
import torch
import random
import warnings
import argparse
import datetime
import shutil
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import json
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
from ObjectDetector.fjn_yolov8 import FJN_YOLOv8 as YOLOv8
warnings.filterwarnings('ignore')



def add_defensivepatch_into_tensor(detector, cfg, img_tensor, dfpatch_tensor, random_palce=False):
    add_canary_clsxywh = []
    for bi in range(img_tensor.shape[0]):
        possible_person = detector.FindHiddenPerson(img_tensor[bi:bi + 1, ::], person_conf=cfg.person_conf, overlap_thresh=cfg.overlap_thresh, faster=cfg.faster)[0]
        tmp_canary_clsxywh = []
        if len(possible_person) > 0:
            for single_possible in possible_person:
                start_x = int((single_possible[0] + single_possible[2]) // 2 - dfpatch_tensor.shape[-2] // 2)
                start_y = int((single_possible[1] + single_possible[3]) // 2 - dfpatch_tensor.shape[-1] // 2)
                location_y, location_x = cfg.defensive_patch_location[0], cfg.defensive_patch_location[1]
                if location_y == 'u':
                    start_y = start_y - 30
                elif location_y == 'c':
                    start_y = start_y
                elif location_y == 'b':
                    start_y = start_y + 30
                else:
                    raise Exception('add_defensive_patch_into_img_tensor: no such defensive_patch_location: y ', cfg.defensive_patch_location)
                if location_x == 'l':
                    start_x = start_x - 30
                elif location_x == 'c':
                    start_x = start_x
                elif location_x == 'r':
                    start_x = start_x + 30
                else:
                    raise Exception('add_defensive_patch_into_img_tensor: no such defensive_patch_location: x ', cfg.defensive_patch_location)
                end_x = int(start_x + dfpatch_tensor.shape[-2])
                end_y = int(start_y + dfpatch_tensor.shape[-1])
                shift_x = random.randint(-10, 10)
                shift_y = random.randint(-10, 10)
                start_y, end_y = start_y + shift_y, end_y + shift_y
                start_x, end_x = start_x + shift_x, end_x + shift_x
                if end_y > img_tensor.shape[2]:
                    end_y = img_tensor.shape[2] - cfg.margin_size
                    start_y = end_y - dfpatch_tensor.shape[-1]
                if end_x > img_tensor.shape[3]:
                    end_x = img_tensor.shape[3] - cfg.margin_size
                    start_x = end_x - dfpatch_tensor.shape[-2]
                if start_y < 0:
                    start_y = cfg.margin_size
                    end_y = dfpatch_tensor.shape[-1] + cfg.margin_size
                if start_x < 0:
                    start_x = cfg.margin_size
                    end_x = dfpatch_tensor.shape[-2] + cfg.margin_size
                img_tensor[bi, :, start_y:end_y, start_x:end_x] = dfpatch_tensor
                tmp_canary_clsxywh.append([cfg.canary_cls_id, (start_x + end_x) / 2 / img_tensor.shape[3],
                                           (start_y + end_y) / 2 / img_tensor.shape[2],
                                           dfpatch_tensor.shape[-2] / img_tensor.shape[3],
                                           dfpatch_tensor.shape[-1] / img_tensor.shape[2]])
            pass
        add_canary_clsxywh.append(tmp_canary_clsxywh)
    return img_tensor, add_canary_clsxywh

    # 返回-图像+数组 将 验证的canary贴在图像上 记录下贴的位置

def reformat_dfpatch_location_xywh(location_xywh, img_sized_with_canary, df_patch):
    location_xyxy = xywh2xyxy(location_xywh)
    start_x, start_y, end_x, end_y = location_xyxy
    if end_y > img_sized_with_canary.shape[0] - 0:
        end_y = img_sized_with_canary.shape[0]
        start_y = end_y - df_patch.shape[0]
    if end_x > img_sized_with_canary.shape[1] :
        end_x = img_sized_with_canary.shape[1]
        start_x = end_x - df_patch.shape[1]
    if start_y < 0:
        start_y = 0
        end_y = df_patch.shape[0]
    if start_x < 0:
        start_x = 0
        end_x = df_patch.shape[1]
    cx, cy, cw, ch = xyxy2xywh( [start_x, start_y, end_x, end_y] )
    return cx, cy, cw, ch

def add_defensivepatch_into_img(cfg, img_sized, dfpatch_cv, possible_person_ls):
    dfpatch_area = []
    if cfg.eval_no_overlap:
        add_box = []
        for idx, item in enumerate(possible_person_ls):
            if np.count_nonzero(item) == 0:
                continue
            if idx == 0:
                add_box.append(item)
            else:
                tmp_is_overlap = False
                for added_item in add_box:
                    item_box = [(item[0] + item[2]) // 2, (item[1] + item[3]) // 2, dfpatch_cv.shape[1], dfpatch_cv.shape[0]]
                    add_item_box = [(added_item[0] + added_item[2]) // 2, (added_item[1] + added_item[3]) // 2, dfpatch_cv.shape[1], dfpatch_cv.shape[0]]
                    item_box[0], item_box[1], item_box[2], item_box[3] = reformat_dfpatch_location_xywh([item_box[0], item_box[1], item_box[2], item_box[3]], img_sized, dfpatch_cv)
                    add_item_box[0], add_item_box[1], add_item_box[2], add_item_box[3] = reformat_dfpatch_location_xywh([add_item_box[0], add_item_box[1], add_item_box[2], add_item_box[3]], img_sized, dfpatch_cv)
                    if tmp_fun_is_overlap(item_box, add_item_box, xyxy=False):
                        tmp_is_overlap = True
                        break
                if not tmp_is_overlap:
                    add_box.append(item)
                pass
            pass
        possible_person_ls = add_box
    for single_person_xyxy in possible_person_ls:
        if np.count_nonzero(single_person_xyxy) == 0:
            continue
        start_x = int((single_person_xyxy[0] + single_person_xyxy[2]) // 2 - dfpatch_cv.shape[1] // 2)
        start_y = int((single_person_xyxy[1] + single_person_xyxy[3]) // 2 - dfpatch_cv.shape[0] // 2)
        location_y, location_x = cfg.defensive_patch_location[0], cfg.defensive_patch_location[1]
        if location_y == 'u':
            start_y = start_y - 30
        elif location_y == 'c':
            start_y = start_y
        elif location_y == 'b':
            start_y = start_y + 30
        else:
            raise Exception('add_defensivepatch_into_img: no such defensive_patch_location: y ', cfg.defensive_patch_location)
        if location_x == 'l':
            start_x = start_x - 30
        elif location_x == 'c':
            start_x = start_x
        elif location_x == 'r':
            start_x = start_x + 30
        else:
            raise Exception('add_defensivepatch_into_img: no such defensive_patch_location: x ', cfg.defensive_patch_location)
        end_x = int(start_x + dfpatch_cv.shape[1])
        end_y = int(start_y + dfpatch_cv.shape[0])
        if end_y > img_sized.shape[0]:
            end_y = img_sized.shape[0] - cfg.margin_size
            start_y = end_y - dfpatch_cv.shape[0]
        if end_x > img_sized.shape[1]:
            end_x = img_sized.shape[1] - cfg.margin_size
            start_x = end_x - dfpatch_cv.shape[1]
        if start_y < 0:
            start_y = cfg.margin_size
            end_y = dfpatch_cv.shape[0] + cfg.margin_size
        if start_x < 0:
            start_x = cfg.margin_size
            end_x = dfpatch_cv.shape[1] + cfg.margin_size
        dfpatch_area.append([start_x, start_y, end_x, end_y])
        img_sized[start_y:end_y, start_x:end_x, :] = dfpatch_cv
    return img_sized, dfpatch_area


def make_yolo_train_label(img_tensor, label_tensor):
    img = img_tensor
    batch_idx = []
    cls = []
    bboxes = []
    for b_idx in range(img.shape[0]):
        for single_label in label_tensor[b_idx]:
            batch_idx.append(b_idx)
            cls.append([single_label[0]])
            if torch.is_tensor(single_label[1:]):
                bboxes.append(single_label[1:].cpu().numpy())
            else:
                bboxes.append(single_label[1:])
    train_batch = {
        'img': img_tensor,
        'batch_idx': torch.Tensor(batch_idx),
        'cls': torch.Tensor(cls),
        'bboxes': torch.Tensor(bboxes)
    }
    return train_batch


class Yolov8Dataset(Dataset):
    def __init__(self, benign_root, adversarial_root, benign_label_root, img_size):
        self.benign_root = benign_root
        self.benign_label_root = benign_label_root
        self.adversarial_root = adversarial_root
        self.img_size = img_size
        self.img_benign_ls = os.listdir(self.benign_root)
        self.img_adversarial_ls = os.listdir(self.adversarial_root)
        assert len(self.img_benign_ls) == len(self.img_adversarial_ls), 'the number of benign images should equal to adversarial images'
        self.img_ls_len = len(self.img_benign_ls)
        self.max_original_label_num = 20

    def __len__(self):
        return self.img_ls_len

    def trans_img_to_tensor(self, img_ori):
        if (img_ori.shape[0] != self.img_size or img_ori.shape[1] != self.img_size):
            img = cv2.resize(img_ori, (self.img_size, self.img_size))
        else:
            img = img_ori
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        return img

    def __getitem__(self, idx):
        img_name = self.img_benign_ls[idx % self.img_ls_len]
        img_benign_path = os.path.join(self.benign_root, img_name)
        img_benign_ori = cv2.imread(img_benign_path, 1)
        img_benign = self.trans_img_to_tensor(img_benign_ori)
        label_benign_txt_name = img_name.split('.')[0] + '.txt'
        label_benign = self.read_benign_label(os.path.join(self.benign_label_root, label_benign_txt_name))
        img_adversarial_path = os.path.join(self.adversarial_root, img_name)
        img_adversarial_ori = cv2.imread(img_adversarial_path, 1)
        img_adversarial = self.trans_img_to_tensor(img_adversarial_ori)
        return img_benign, label_benign, img_adversarial

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


class Canary:

    def __init__(self, cfg, detector):
        self.cfg = cfg
        self.detector = detector
        self.img_size = self.cfg.img_size
        self.person_conf = self.cfg.person_conf
        self.overlap_thresh = self.cfg.overlap_thresh
        self.data_loader = None
        self.canary_tensor = None
        self.canary_save_path = ''
        self.eval_canary = None
        self.canary_size = self.cfg.canary_size
        self.canary_cls_id = self.cfg.canary_cls_id
        self.eval_no_overlap = self.cfg.eval_no_overlap
        self.margin_size = self.cfg.margin_size
        self.faster = False
        pass

    def train(self):
        optimizer = optim.Adam([self.canary_tensor], lr=self.cfg.learing_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epoch)
        t = tqdm(total=self.cfg.epoch, ascii=True)
        for epoch in range(1, self.cfg.epoch + 1):
            t.set_description((f'canary_{self.cfg.weight} epoch: {epoch}/{self.cfg.epoch}'))
            for i_batch, (benign_input_tensor, benign_label, adv_input_tensor) in enumerate(self.data_loader):
                benign_input_tensor = benign_input_tensor.requires_grad_(False).cuda()
                adv_input_tensor = adv_input_tensor.requires_grad_(False).cuda()
                self.detector.model.eval()
                adv_input_canary_tensor, adv_canary_clsxywh = add_defensivepatch_into_tensor(self.detector, self.cfg, adv_input_tensor, self.canary_tensor, random_palce=True)
                benign_input_canary_tensor, benign_canary_clsxywh = add_defensivepatch_into_tensor(self.detector, self.cfg, benign_input_tensor, self.canary_tensor, random_palce=True)
                adv_train_batch = make_yolo_train_label(adv_input_canary_tensor, adv_canary_clsxywh)
                benign_train_batch = make_yolo_train_label(benign_input_canary_tensor, benign_canary_clsxywh)
                self.detector.model.train()
                adv_loss, adv_loss_items = self.detector.model.model(adv_train_batch)
                benign_loss, benign_loss_items = self.detector.model.model(benign_train_batch)
                loss = self.cfg.weight * benign_loss - adv_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.canary_tensor.requires_grad_(True)
                self.canary_tensor.data.clamp_(0.0001, 1)
                t.set_postfix({
                    'b_box': '{0:1.4f}'.format(benign_loss_items[0].cpu().item()),
                    'b_cls': '{0:1.4f}'.format(benign_loss_items[1].cpu().item()),
                    'b_dfl': '{0:1.4f}'.format(benign_loss_items[2].cpu().item()),
                    'b_loss': '{0:1.4f}'.format(benign_loss.cpu().item()),
                    'a_box': '{0:1.4f}'.format(adv_loss_items[0].cpu().item()),
                    'a_cls': '{0:1.4f}'.format(adv_loss_items[1].cpu().item()),
                    'a_dfl': '{0:1.4f}'.format(adv_loss_items[2].cpu().item()),
                    'a_loss': '{0:1.4f}'.format(adv_loss.cpu().item())
                })
            if epoch % self.cfg.epoch_save == 0:
                self.save_canary(epoch)
            scheduler.step()
            t.update(1)
        t.close()
        pass

    def init_dataloader(self):
        train_dataset = Yolov8Dataset(self.cfg.benign_root, self.cfg.adversarial_root, self.cfg.benign_label_root, self.img_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_works, drop_last=True)
        self.data_loader = train_loader
        pass

    def init_canary(self):
        if self.cfg.canary_init and os.path.exists(os.path.join(self.cfg.canary_init_path, f'{self.cfg.canary_cls_id}.jpg')):
            canary_img = cv2.imread(os.path.join(self.cfg.canary_init_path, f'{self.cfg.canary_cls_id}.jpg'), 1)
            canary_img = cv2.cvtColor(canary_img, cv2.COLOR_BGR2RGB)
            canary_img = cv2.resize(canary_img, (self.canary_size, self.canary_size))
            canary_img = np.transpose(canary_img, (2, 0, 1)) / 255.
        else:
            canary_img = np.ones((3, self.canary_size, self.canary_size)) * 0.5
        self.canary_img = canary_img
        canary_tensor = torch.from_numpy(canary_img)
        self.canary_tensor = canary_tensor.requires_grad_(True)
        self.canary_cls_id = self.cfg.canary_cls_id
        fold_name = f'canary_{self.cfg.defensive_patch_location}_{self.cfg.attack_name}_{self.cfg.detect_name}_{self.cfg.weight}_{self.cfg.batch_size}'
        canary_fold = f'exp_{self.cfg.data_name}_{self.cfg.canary_cls_id}_{self.canary_size}_{self.cfg.epoch}'
        self.canary_save_path = os.path.join('./FJNTraining', fold_name, canary_fold)
        os.makedirs(self.canary_save_path, exist_ok=True)
        del canary_img, fold_name, canary_fold
        self.save_canary(0)
        pass

    def save_canary(self, epoch):
        canary_img_RGB = transforms.ToPILImage('RGB')(self.canary_tensor)
        canary_img_RGB = np.array(canary_img_RGB).astype(np.uint8)
        canary_img_BGR = cv2.cvtColor(canary_img_RGB, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(f'{self.canary_save_path}', f'canary_{str(epoch).zfill(3)}.png'), canary_img_BGR)
        del canary_img_BGR
        pass

    def eval_single(self, img_cv_ori):
        img_cv = copy.deepcopy(img_cv_ori)
        possible_person_ls = self.detector.FindHiddenPerson(deepcopy(img_cv), person_conf=self.person_conf, overlap_thresh=self.overlap_thresh, faster=self.faster, remove_small_length=20, shown=shown)[0]
        if (img_cv.shape[0] != self.img_size or img_cv.shape[1] != self.img_size):
            img_sized = cv2.resize(img_cv, (self.cfg.img_size, self.cfg.img_size))
        else:
            img_sized = img_cv

        if len(possible_person_ls) == 0 or (np.count_nonzero(possible_person_ls) == 0):
            is_attack = False
        else:
            img_original_result = self.detector.detect_single(img_cv)[0]
            if img_original_result is not None:
                original_canary_result = img_original_result[np.where(img_original_result[:, -1] == self.canary_cls_id)]
                canary_original_num = len(original_canary_result)
            else:
                canary_original_num = 0
            img_sized_with_canary = deepcopy(img_sized)
            img_sized_with_canary, canary_area = add_defensivepatch_into_img(self.cfg, img_sized_with_canary, self.eval_canary, possible_person_ls)
            canary_put_num = len(canary_area)
            img_with_canary_result = self.detector.detect_single(img_sized_with_canary)[0]
            if img_with_canary_result is not None:
                canary_result = img_with_canary_result[np.where(img_with_canary_result[:, -1] == self.canary_cls_id)]
                canary_detected_num = len(canary_result)
            else:
                canary_detected_num = 0
            if canary_original_num + canary_put_num == canary_detected_num:
                is_attack = False
            else:
                is_attack = True
        return is_attack

    def eval_load_canary(self, canary_path, canary_cls_id):
        canary_img = cv2.imread(canary_path, 1)
        if canary_img.shape[0] != self.canary_size and canary_img.shape[1] != self.canary_size:
            canary_img = cv2.resize(canary_img, (self.canary_size, self.canary_size))
        self.eval_canary = canary_img
        self.canary_cls_id = canary_cls_id
        print(f'Load canary({canary_cls_id}) for eval: {canary_path}')
        pass

def xyxy2xywh(single_box):
    res = [0, 0, 0, 0]
    res[0] = (single_box[0] + single_box[2]) / 2
    res[1] = (single_box[1] + single_box[3]) / 2
    res[2] = (single_box[2] - single_box[0])
    res[3] = (single_box[3] - single_box[1])
    return res

def xywh2xyxy(single_box):
    res = [0, 0, 0, 0]
    res[0] = single_box[0] - single_box[2] / 2
    res[2] = single_box[0] + single_box[2] / 2
    res[1] = single_box[1] - single_box[3] / 2
    res[3] = single_box[1] + single_box[3] / 2
    return res

def tmp_fun_is_overlap(box1, box2, xyxy=True):
    if not xyxy:
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)
    start_x_1, start_y_1, end_x_1, end_y_1 = box1
    start_x_2, start_y_2, end_x_2, end_y_2 = box2
    return not ((start_x_1 > end_x_2) or (start_x_2 > end_x_1) or (start_y_1 > end_y_2) or (start_y_2 > end_y_1))


class Woodpecker:
    def __init__(self, cfg, detector):
        self.cfg = cfg
        self.detector = detector
        self.img_size = self.cfg.img_size
        self.person_conf = self.cfg.person_conf
        self.overlap_thresh = self.cfg.overlap_thresh
        self.data_loader = None
        self.wd_tensor = None
        self.wd_save_path = ''
        self.eval_wd = None
        self.wd_size = self.cfg.wd_size
        self.eval_no_overlap = self.cfg.eval_no_overlap
        self.margin_size = self.cfg.margin_size
        self.faster = False
        pass

    def train(self):
        optimizer = optim.Adam([self.wd_tensor], lr=self.cfg.learing_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epoch)
        t = tqdm(total=self.cfg.epoch, ascii=True)
        for epoch in range(1, self.cfg.epoch + 1):
            t.set_description((f'wd_{self.cfg.weight} epoch: {epoch}/{self.cfg.epoch}'))
            for i_batch, (benign_input_tensor, benign_label, adv_input_tensor) in enumerate(self.data_loader):
                benign_input_tensor = benign_input_tensor.requires_grad_(False).cuda()
                adv_input_tensor = adv_input_tensor.requires_grad_(False).cuda()
                self.detector.model.eval()
                adv_input_wd_tensor, _ = add_defensivepatch_into_tensor(self.detector, self.cfg, adv_input_tensor, self.wd_tensor, random_palce=True)
                benign_input_wd_tensor, _ = add_defensivepatch_into_tensor(self.detector, self.cfg, benign_input_tensor, self.wd_tensor, random_palce=True)
                adv_train_batch = make_yolo_train_label(adv_input_wd_tensor, benign_label)
                benign_train_batch = make_yolo_train_label(benign_input_wd_tensor, benign_label)
                self.detector.model.train()
                adv_loss, adv_loss_items = self.detector.model.model(adv_train_batch)
                benign_loss, benign_loss_items = self.detector.model.model(benign_train_batch)
                loss = self.cfg.weight * benign_loss + adv_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.wd_tensor.requires_grad_(True)
                self.wd_tensor.data.clamp_(0.0001, 1)
                t.set_postfix({
                    'b_box': '{0:1.4f}'.format(benign_loss_items[0].cpu().item()),
                    'b_cls': '{0:1.4f}'.format(benign_loss_items[1].cpu().item()),
                    'b_dfl': '{0:1.4f}'.format(benign_loss_items[2].cpu().item()),
                    'b_loss': '{0:1.4f}'.format(benign_loss.cpu().item()),
                    'a_box': '{0:1.4f}'.format(adv_loss_items[0].cpu().item()),
                    'a_cls': '{0:1.4f}'.format(adv_loss_items[1].cpu().item()),
                    'a_dfl': '{0:1.4f}'.format(adv_loss_items[2].cpu().item()),
                    'a_loss': '{0:1.4f}'.format(adv_loss.cpu().item())
                })
            if epoch % self.cfg.epoch_save == 0:
                self.save_wd(epoch)
            scheduler.step()
            t.update(1)
        t.close()
        pass

    def init_dataloader(self):
        train_dataset = Yolov8Dataset(self.cfg.benign_root, self.cfg.adversarial_root, self.cfg.benign_label_root, self.img_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_works, drop_last=True)
        self.data_loader = train_loader
        pass

    def save_wd(self, epoch):
        wd_img_RGB = transforms.ToPILImage('RGB')(self.wd_tensor)
        wd_img_RGB = np.array(wd_img_RGB).astype(np.uint8)
        wd_img_BGR = cv2.cvtColor(wd_img_RGB, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(f'{self.wd_save_path}', f'wd_{str(epoch).zfill(3)}.png'), wd_img_BGR)
        del wd_img_BGR
        pass

    def eval_load_wd(self, wd_path):
        wd_img = cv2.imread(wd_path, 1)
        if wd_img.shape[0] != self.wd_size and wd_img.shape[1] != self.wd_size:
            wd_img = cv2.resize(wd_img, (self.wd_size, self.wd_size))
        self.eval_wd = wd_img
        print(f'Load Woodpecker for eval: {wd_path}')
        pass

    def init_woodpecker(self):
        wd_img = np.random.randint(0, 255, (3, self.wd_size, self.wd_size)) / 255.
        wd_tensor = torch.from_numpy(wd_img)
        self.wd_tensor = wd_tensor.requires_grad_(True)
        fold_name = f'wd_{self.cfg.defensive_patch_location}_{self.cfg.attack_name}_{self.cfg.detect_name}_{self.cfg.weight}_{self.cfg.batch_size}'
        wd_fold = f'exp_{self.cfg.data_name}_{self.wd_size}_{self.cfg.epoch}'
        self.wd_save_path = os.path.join('./FJNTraining', fold_name, wd_fold)
        os.makedirs(self.wd_save_path, exist_ok=True)
        self.save_wd(0)
        del wd_img
        pass

    def eval_single(self, img_cv):
        possible_person_ls = self.detector.FindHiddenPerson(deepcopy(img_cv), person_conf=self.person_conf, overlap_thresh=self.overlap_thresh, faster=self.faster, remove_small_length=20, shown=shown)[0]
        if (img_cv.shape[0] != self.img_size or img_cv.shape[1] != self.img_size):
            img_sized = cv2.resize(img_cv, (self.cfg.img_size, self.cfg.img_size))
        else:
            img_sized = img_cv
        if len(possible_person_ls) == 0 or (np.count_nonzero(possible_person_ls) == 0):
            return False
        else:
            img_original_result = self.detector.detect_single(img_sized)[0]
            if img_original_result is not None:
                detected_person_in_original_ls = img_original_result[np.where(img_original_result[:, -1] == 0)]
            else:
                detected_person_in_original_ls = []
            img_sized_with_wd = deepcopy(img_sized)
            img_sized_with_wd, _ = add_defensivepatch_into_img(self.cfg, img_sized_with_wd, self.eval_wd, possible_person_ls)
            img_with_wd_result = self.detector.detect_single(img_sized_with_wd)[0]
            if img_with_wd_result is not None:
                detected_person_in_wd_ls = img_with_wd_result[np.where(img_with_wd_result[:, -1] == 0)]
            else:
                detected_person_in_wd_ls = []
            has_new_person = False
            if len(detected_person_in_original_ls) == 0 and len(detected_person_in_wd_ls) == 0:
                has_new_person = False
            elif len(detected_person_in_original_ls) == 0 and len(detected_person_in_wd_ls) > 0:
                has_new_person = True
            else:
                for new_single_person in detected_person_in_wd_ls:
                    temp_ls = []
                    for old_single_person in detected_person_in_original_ls:
                        chongdie = self.cal_chongdie(new_single_person[:4], old_single_person[:4], x1y1x2y2=True)
                        temp_ls.append(chongdie)
                    if max(temp_ls) < 0.4:
                        has_new_person = True
                        break
            return has_new_person

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


def freeze_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    Gparser = argparse.ArgumentParser(description='Combination of Canary and Woodpecker')
    Gparser.add_argument('--detect_name', default='yolov8', type=str, help='')
    Gparser.add_argument('--attack_name', default='fool', type=str, help='')
    Gparser.add_argument('--weight', default=1.0, type=float, help='weight of benign loss')
    Gparser.add_argument('--epoch', default=50, type=int, help='epoch for train')
    Gparser.add_argument('--learing_rate', default=0.05, type=float, help='batch_size for training')
    Gparser.add_argument('--epoch_save', default=1, type=int, help='epoch for save model and canary')
    Gparser.add_argument('--img_size', default=640, type=int)
    Gparser.add_argument('--data_name', default='VOC07_120', type=str, help='')
    Gparser.add_argument('--benign_root', default="Data/traineval/VOC07_YOLOv8/train_120/benign", type=str)
    Gparser.add_argument('--benign_label_root', default="Data/traineval/VOC07_YOLOv8/train_120/benign_label", type=str)
    Gparser.add_argument('--adversarial_root', default="Data/traineval/VOC07_YOLOv8/train_120/adversarial", type=str)
    Gparser.add_argument('--batch_size', default=10, type=int, help='batch_size for training')
    Gparser.add_argument('--num_works', default=5, type=int, help='num_works')
    Gparser.add_argument('--margin_size', default=0, type=int, help='margin size')
    Gparser.add_argument('--person_conf', default=0.05, type=float, help='person_conf')
    Gparser.add_argument('--defensive_patch_location', default='cc', type=str, help='defensive patch location', choices=['uc', 'cl', 'cr', 'bc', 'cc'])
    Gparser.add_argument('--eval_no_overlap', action='store_true', default=True, help='eval_no_overlap')
    Gparser.add_argument('--overlap_thresh', default=0.4, type=float, help='overlap_thresh')
    Gparser.add_argument('--canary_init_path', default='Data/InitImages/', type=str, help='canary init image')
    Gparser.add_argument('--canary_init', action='store_true', default=True, help='options :True or False')
    Gparser.add_argument('--canary_cls_id', default=22, type=int, help='canary label')
    Gparser.add_argument('--canary_size', default=80, type=int, help='canary size')
    Gparser.add_argument('--wd_size', default=80, type=int, help='woodpecker size')
    Gparser.add_argument('--shuffle', action='store_true', default=True, help='options :True or False')
    Gparser.add_argument('--max_label', default=20, type=int, help='max_label')
    Gparser.add_argument('--seed', default=301, type=int, help='choose seed')
    Gparser.add_argument('--faster', action='store_true', default=False, help='faster')
    Gparser.add_argument('--gpu_id', default=0, type=int, help='choose gpu')
    Gparser.add_argument('--train', action='store_true', help='train')
    Gparser.add_argument('--test', action='store_true', help='test')
    Gparser.add_argument('--df_mode', default='C', type=str, help='select df_patch', choices=['C', 'W', 'A'])
    Gparser.add_argument('--input_img', default='', type=str, help='the patch of input img')
    Gparser.add_argument('--best_canary_path', default='./trained_dfpatches/YOLOv8/canary.png', type=str, help='the patch of best_canary')
    Gparser.add_argument('--best_wd_path', default='./trained_dfpatches/YOLOv8/wd.png', type=str, help='the patch of best_wd')

    return Gparser.parse_known_args()[0]


'''
python YOLOv8_Combiner.py --train --df_mode C --defensive_patch_location cc --canary_cls_id 22 --canary_size 80 --person_conf 0.05 --weight 2.0
python YOLOv8_Combiner.py --train --df_mode W --defensive_patch_location cc --wd_size 80 --person_conf 0.05 --weight 1.0

python YOLOv8_Combiner.py --test --df_mode C --defensive_patch_location cc --canary_cls_id 22 --canary_size 80 --person_conf 0.05 --best_canary_path ./trained_dfpatches/YOLOv8/canary.png --input_img XXX
python YOLOv8_Combiner.py --test --df_mode W --defensive_patch_location cc --wd_size 80 --person_conf 0.05 --best_wd_path ./trained_dfpatches/YOLOv8/wd.png --input_img XXX
python YOLOv8_Combiner.py --test --df_mode A --defensive_patch_location cc --canary_cls_id 22 --canary_size 80 --wd_size 80 --person_conf 0.05 --best_canary_path ./trained_dfpatches/YOLOv8/canary.png --best_wd_path ./trained_dfpatches/YOLOv8/wd.png --input_img XXX
'''


if __name__ == '__main__':
    cfg = get_args()
    freeze_seed(cfg.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.gpu_id}'
    
    detector = YOLOv8()
    
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
            canary.eval_load_canary(canary_path=cfg.best_canary_path, canary_cls_id=cfg.canary_cls_id)
        if cfg.df_mode == 'W' or cfg.df_mode == 'A' :
            woodpecker = Woodpecker(cfg, detector)
            woodpecker.eval_load_wd(wd_path=cfg.best_wd_path)
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

