import os
import cv2
import torch
import random
import warnings
import argparse
import datetime
import shutil
import copy
import json

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from ObjectDetector.fjn_fasterrcnn import fjn_faster_rcnn

warnings.filterwarnings('ignore')



class CanaryFasterRCNNDataset(Dataset):
    def __init__(self, clean_root, attack_root, clean_hidden_root, attack_hidden_root, canary_size, canary_cls_id):
        self.clean_root = clean_root
        self.clean_hidden_root = clean_hidden_root
        self.attack_root = attack_root
        self.attack_hidden_root = attack_hidden_root
        self.canary_size = canary_size
        self.canary_cls_id = canary_cls_id
        self.img_clean_ls = os.listdir(self.clean_root)
        self.img_attack_ls = os.listdir(self.attack_root)
        self.img_ls_len = len(self.img_clean_ls)

    def __len__(self):
        return self.img_ls_len

    def __getitem__(self, idx):
        clean_img_name = self.img_clean_ls[idx % self.img_ls_len]
        attack_img_name = self.img_attack_ls[idx % self.img_ls_len]
        
        img_clean_path = os.path.join(self.clean_root, clean_img_name)
        img_clean_input = cv2.imread(img_clean_path, 1)
        img_clean_input = cv2.cvtColor(img_clean_input, cv2.COLOR_BGR2RGB)
        img_clean_tensor = torch.from_numpy(img_clean_input.transpose(2, 0, 1)).float().div(255.0)
        
        label_clean_path = os.path.join(self.clean_hidden_root, os.path.splitext(clean_img_name)[0] + '.txt')
        label_img_clean = np.loadtxt(label_clean_path)
        label_img_clean = self.process_lab(label_img_clean)
        
        img_attack_path = os.path.join(self.attack_root, attack_img_name)
        img_attack_input = cv2.imread(img_attack_path, 1)
        img_attack_input = cv2.cvtColor(img_attack_input, cv2.COLOR_BGR2RGB)
        img_attack_tensor = torch.from_numpy(img_attack_input.transpose(2, 0, 1)).float().div(255.0)
        
        label_attack_path = os.path.join(self.attack_hidden_root, os.path.splitext(attack_img_name)[0] + '.txt')
        label_img_attack = np.loadtxt(label_attack_path)
        label_img_attack = self.process_lab(label_img_attack)
        
        return img_clean_tensor, img_attack_tensor, label_img_clean, label_img_attack

    def process_lab(self, lab):
        label = lab
        if len(lab):
            if len(label.shape) == 1:
                label = label[np.newaxis, :]
            label[:, 0] = self.canary_cls_id
            label[:,1], label[:,3] = (label[:,1] + label[:,3])//2 - self.canary_size // 2, (label[:,1] + label[:,3])//2 + self.canary_size // 2
            label[:,2], label[:,4] = (label[:,2] + label[:,4])//2 - self.canary_size // 2, (label[:,2] + label[:,4])//2 + self.canary_size // 2
        return label


class Canary:
    def __init__(self, cfg, detector):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.detector = detector
        self.person_conf = cfg.person_conf
        self.overlap_thresh = cfg.overlap_thresh
        self.data_loader = None
        self.canary_tensor = None
        self.canary_save_path = ''
        self.eval_canary = None
        self.canary_cls_id = -1
        pass

    def train(self):
        optimizer = optim.Adam([self.canary_tensor], lr=self.cfg.learing_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epoch)
        self.detector.detector.train()
        t = tqdm(total=self.cfg.epoch, ascii=True)
        for epoch in range(1, cfg.epoch + 1):
            t.set_description((f'canary_{self.cfg.attack_name}_{self.cfg.detect_name}_{cfg.weight} epoch: {epoch}/{self.cfg.epoch}'))

            for i_batch, (benign_input_tensor, attack_input_tensor, img_benign_label, img_label_attack) in enumerate(self.data_loader):
                benign_input_tensor = benign_input_tensor.requires_grad_(False).to(self.device)
                attack_input_tensor = attack_input_tensor.requires_grad_(False).to(self.device)

                benign_input_canary_tensor = self.add_canary_into_img_tensor(benign_input_tensor, img_benign_label)
                attack_input_canary_tensor = self.add_canary_into_img_tensor(attack_input_tensor, img_label_attack)
                
                loss_all = 0

                if img_benign_label.shape[1] > 0:
                    benign_target = [{"boxes": ssl[:, 1:5].to(self.device), "labels":ssl[:,0].long().to(self.device)  } for ssl in img_benign_label]
                    loss_dict_benign = self.detector.detector(benign_input_canary_tensor, benign_target)
                    losses_benign = sum(loss for loss in loss_dict_benign.values())
                    loss_all = loss_all + self.cfg.weight * losses_benign

                if img_label_attack.shape[1] > 0:
                    adv_attack_target = [{"boxes": ssl[:, 1:5].to(self.device), "labels":ssl[:,0].long().to(self.device)  } for ssl in img_label_attack]
                    loss_dict_adv_attack = self.detector.detector(attack_input_canary_tensor, adv_attack_target)
                    losses_adv_attack = sum(loss for loss in loss_dict_adv_attack.values())
                    loss_all = loss_all - losses_adv_attack

                if loss_all == 0:
                    continue
                
                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()

                self.canary_tensor.requires_grad_(True)
                self.canary_tensor.data.clamp_(0, 1)
                t.set_postfix({
                    'loss_all': '{0:1.5f}'.format(loss_all / len(self.data_loader))
                })
            if epoch % self.cfg.epoch_save == 0:
                self.save_canary(epoch)
            t.update(1)
            scheduler.step()
        t.close()
        pass

    def add_canary_into_img_tensor(self, img_tensor, possible_person):
        for bi in range(img_tensor.shape[0]):
            for single_xyxy in possible_person[bi]:
                start_x = int(single_xyxy[1].detach().item())
                start_y = int(single_xyxy[2].detach().item())
                end_x = int(single_xyxy[3].detach().item())
                end_y = int(single_xyxy[4].detach().item())
                cx = (start_x + end_x) // 2
                cy = (start_y + end_y) // 2
                start_x, start_y = cx - self.canary_tensor.shape[-1] // 2, cy - self.canary_tensor.shape[-2] // 2
                end_x, end_y = start_x + self.canary_tensor.shape[-1], start_y + self.canary_tensor.shape[-2]
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
                if end_y > img_tensor.shape[2]:
                    end_y = img_tensor.shape[2]
                    start_y = end_y - self.canary_tensor.shape[-1]
                if end_x > img_tensor.shape[3]:
                    end_x = img_tensor.shape[3]
                    start_x = end_x - self.canary_tensor.shape[-2]
                if start_y < 0:
                    start_y = 0
                    end_y = self.canary_tensor.shape[-1]
                if start_x < 0:
                    start_x = 0
                    end_x = self.canary_tensor.shape[-2]
                img_tensor[bi, :, start_y:end_y, start_x:end_x] = self.canary_tensor
        return img_tensor

    def save_canary(self, epoch):
        rap_img = transforms.ToPILImage('RGB')(self.canary_tensor)
        rap_img = np.array(rap_img).astype(np.uint8)
        rap_img = cv2.cvtColor(rap_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(f'{self.canary_save_path}', f'canary_{str(epoch).zfill(3)}.png'), rap_img)
        del rap_img
        pass

    def eval_single(self, img_cv):
        self.detector.detector.eval()
        img_original_result, possible_person_ls = self.detector.detect_for_hidden_person(img_cv, interest_score_thresh=self.person_conf)
        if len(possible_person_ls) == 0:
            return False
        else:
            if len(img_original_result) and len(img_original_result[0]) > 0:
                original_canary_result = img_original_result[np.where(img_original_result[:, -1] == self.canary_cls_id)]
            else:
                original_canary_result = []
            img_sized_with_canary = copy.deepcopy(img_cv)
            img_sized_with_canary, canary_area = self.eval_add_canary_into_img(img_sized_with_canary, possible_person_ls)
            img_with_canary_result = self.detector.detect_single_img_cv(img_sized_with_canary)
            if len(img_with_canary_result)  and len(img_with_canary_result[0])>0:
                canary_result = img_with_canary_result[np.where(img_with_canary_result[:, -1] == self.canary_cls_id)]
            else:
                canary_result = []
            canary_original_num = len(original_canary_result)
            canary_put_num = len(canary_area)
            canary_detected_num = len(canary_result)
            if canary_original_num + canary_put_num == canary_detected_num:
                return False
            else:
                return True

    def eval_add_canary_into_img(self, img_sized_with_canary, possible_person_ls):
        canary_area = []
        add_eval_caanry = copy.deepcopy(self.eval_canary)
        for single_person_xyxy in possible_person_ls:
            start_x = int(single_person_xyxy[0])
            start_y = int(single_person_xyxy[1])
            end_x = int(single_person_xyxy[2])
            end_y = int(single_person_xyxy[3])
            cx = (start_x + end_x) // 2
            cy = (start_y + end_y) // 2
            start_x, start_y = cx - add_eval_caanry.shape[1] // 2, cy - add_eval_caanry.shape[0] // 2
            end_x, end_y = start_x + add_eval_caanry.shape[1], start_y + add_eval_caanry.shape[0]
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
            if end_y > img_sized_with_canary.shape[0]:
                end_y = img_sized_with_canary.shape[0]
                start_y = end_y - add_eval_caanry.shape[0]
            if end_x > img_sized_with_canary.shape[1]:
                end_x = img_sized_with_canary.shape[1]
                start_x = end_x - add_eval_caanry.shape[1]
            if start_y < 0:
                start_y = 0
                end_y = add_eval_caanry.shape[0]
            if start_x < 0:
                start_x = 0
                end_x = add_eval_caanry.shape[1]
            canary_area.append([start_x, start_y, end_x, end_y])
            img_sized_with_canary[start_y:end_y, start_x:end_x, :] = add_eval_caanry
        return img_sized_with_canary, canary_area

    def eval_load_canary(self, canary_path, canary_cls_id):
        canary_img = cv2.imread(canary_path, 1)
        if (canary_img.shape[0] != self.cfg.canary_size) or (canary_img.shape[1] != self.cfg.canary_size):
            canary_img = cv2.resize(canary_img, (self.cfg.canary_size, self.cfg.canary_size))
        self.eval_canary = canary_img
        self.canary_cls_id = canary_cls_id
        print(f'Load canary({canary_cls_id}) for eval: {canary_path}')
        pass

    def init_dataloader(self):
        train_dataset = CanaryFasterRCNNDataset(self.cfg.clean_root, self.cfg.attack_root, self.cfg.clean_hidden_root, self.cfg.attack_hidden_root, self.cfg.canary_size, self.cfg.canary_cls_id)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_works)
        self.data_loader = train_loader
        pass

    def init_canary(self):
        if self.cfg.canary_init and os.path.exists( os.path.join(self.cfg.canary_init_path, f'{self.cfg.canary_cls_id}.jpg') ):
            canary_img = cv2.imread(os.path.join(self.cfg.canary_init_path, f'{self.cfg.canary_cls_id}.jpg'), 1)
            canary_img = cv2.resize(canary_img, (self.cfg.canary_size, self.cfg.canary_size))
            canary_img = np.transpose(canary_img, (2, 0, 1)) / 255.
        else:
            canary_img = np.random.randint(0, 255, (3, self.cfg.canary_size, self.cfg.canary_size)) / 255.
        canary_tensor = torch.from_numpy(canary_img)
        self.canary_tensor = canary_tensor.requires_grad_(True)
        fold_name = f'canary_{self.cfg.defensive_patch_location}_{self.cfg.attack_name}_{self.cfg.detect_name}_{self.cfg.weight}_{self.cfg.batch_size}'
        canary_fold = f'exp_{self.cfg.data_name}_{self.cfg.canary_cls_id}_{self.cfg.canary_size}'
        self.canary_save_path = os.path.join('./FJNTraining',  fold_name, canary_fold)
        os.makedirs(self.canary_save_path, exist_ok=True)
        del canary_img, fold_name, canary_fold
        self.save_canary(0)
        pass


class WoodpeckerFasterRCNNDataset(Dataset):
    def __init__(self, clean_root, attack_root, clean_hidden_root, attack_hidden_root, attack_label_root, wd_size):
        self.clean_root = clean_root
        self.attack_root = attack_root
        self.clean_hidden_root = clean_hidden_root
        self.attack_hidden_root = attack_hidden_root
        self.attack_label_root = attack_label_root
        self.wd_size = wd_size
        self.img_clean_ls = os.listdir(self.clean_root)
        self.img_attack_ls = os.listdir(self.attack_root)
        self.img_ls_len = len(self.img_attack_ls)

    def __len__(self):
        return self.img_ls_len

    def __getitem__(self, idx):
        clean_img_name = self.img_clean_ls[idx % self.img_ls_len]
        attack_img_name = self.img_attack_ls[idx % self.img_ls_len]
        
        img_clean_path = os.path.join(self.clean_root, clean_img_name)
        img_clean_input = cv2.imread(img_clean_path, 1)
        img_clean_input = cv2.cvtColor(img_clean_input, cv2.COLOR_BGR2RGB)
        img_clean_tensor = torch.from_numpy(img_clean_input.transpose(2, 0, 1)).float().div(255.0)
        
        label_clean_hidden_path = os.path.join(self.clean_hidden_root, os.path.splitext(clean_img_name)[0] + '.txt')
        label_clean_hidden = np.loadtxt(label_clean_hidden_path)
        label_clean_hidden = self.process_lab(label_clean_hidden)
        
        label_clean_path = os.path.join(self.attack_label_root, os.path.splitext(clean_img_name)[0] + '.txt')
        label_clean = np.loadtxt(label_clean_path)
        label_clean = self.process_lab(label_clean)
        
        img_attack_path = os.path.join(self.attack_root, attack_img_name)
        img_attack_input = cv2.imread(img_attack_path, 1)
        img_attack_input = cv2.cvtColor(img_attack_input, cv2.COLOR_BGR2RGB)
        img_attack_tensor = torch.from_numpy(img_attack_input.transpose(2, 0, 1)).float().div(255.0)
        
        label_attack_hidden_path = os.path.join(self.attack_hidden_root, os.path.splitext(attack_img_name)[0] + '.txt')
        label_attack_hidden = np.loadtxt(label_attack_hidden_path)
        label_attack_hidden = self.process_lab(label_attack_hidden)
        
        return img_clean_tensor, img_attack_tensor, label_clean_hidden, label_attack_hidden, label_clean

    def process_lab(self, lab):
        label = lab
        if len(lab):
            if len(label.shape) == 1:
                label = label[np.newaxis, :]
            label[:,1], label[:,3] = (label[:,1] + label[:,3])//2 - self.wd_size // 2, (label[:,1] + label[:,3])//2 + self.wd_size // 2
            label[:,2], label[:,4] = (label[:,2] + label[:,4])//2 - self.wd_size // 2, (label[:,2] + label[:,4])//2 + self.wd_size // 2
        return label


class Woodpecker:
    def __init__(self, cfg, detector):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.detector = detector
        self.person_conf = cfg.person_conf
        self.overlap_thresh = cfg.overlap_thresh
        self.data_loader = None
        self.wood_tensor = None
        self.wood_save_path = ''
        self.eval_wd = None
        pass
    
    def train(self):
        optimizer = optim.Adam([self.wd_tensor], lr=self.cfg.learing_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epoch)
        self.detector.detector.train()
        t = tqdm(total=self.cfg.epoch, ascii=True)
        for epoch in range(1, cfg.epoch + 1):
            t.set_description((f'wd_{self.cfg.attack_name}_{self.cfg.detect_name}_{cfg.weight} epoch: {epoch}/{self.cfg.epoch}'))
            for i_batch, (img_benign_tensor, img_attack_tensor, label_benign_hidden, label_attack_hidden, label_benign) in enumerate(self.data_loader):
                
                attack_input_tensor = img_attack_tensor.requires_grad_(False).to(self.device)
                attack_input_wd_tensor = self.add_wd_into_img_tensor(attack_input_tensor, label_attack_hidden)
                
                benign_input_tensor = img_benign_tensor.requires_grad_(False).to(self.device)
                benign_input_wd_tensor = self.add_wd_into_img_tensor(benign_input_tensor, label_benign_hidden)
                
                loss_all = 0
                
                if label_benign_hidden.shape[1] > 0:
                    benign_target = [{"boxes": ssl[:, 1:5].to(self.device), "labels":ssl[:,0].long().to(self.device)  } for ssl in label_benign]
                    loss_dict_benign = self.detector.detector(benign_input_wd_tensor, benign_target)
                    losses_benign = sum(loss for loss in loss_dict_benign.values())
                    loss_all = loss_all + self.cfg.weight * losses_benign

                if label_attack_hidden.shape[1] > 0:
                    adv_attack_target = [{"boxes": ssl[:, 1:5].to(self.device), "labels":ssl[:,0].long().to(self.device)  } for ssl in label_benign]
                    loss_dict_adv_attack = self.detector.detector(attack_input_wd_tensor, adv_attack_target)
                    losses_adv_attack = sum(loss for loss in loss_dict_adv_attack.values())
                    loss_all = loss_all + losses_adv_attack

                if loss_all == 0:
                    continue
                
                optimizer.zero_grad()
                loss_all.backward(retain_graph=True)
                optimizer.step()
                self.wd_tensor.requires_grad_(True)
                self.wd_tensor.data.clamp_(0, 1)
                t.set_postfix({
                    'loss_all': '{0:1.5f}'.format(loss_all / len(self.data_loader))
                })
            if epoch % self.cfg.epoch_save == 0:
                self.save_wd(epoch)
            t.update(1)
            scheduler.step()
        t.close()
        pass

    def add_wd_into_img_tensor(self, img_tensor, possible_person):
        for bi in range(img_tensor.shape[0]):
            for single_xyxy in possible_person[bi]:
                start_x = int(single_xyxy[1].detach().item())
                start_y = int(single_xyxy[2].detach().item())
                end_x = int(single_xyxy[3].detach().item())
                end_y = int(single_xyxy[4].detach().item())
                cx = (start_x + end_x) // 2
                cy = (start_y + end_y) // 2
                start_x, start_y = cx - self.wd_tensor.shape[-1] // 2, cy - self.wd_tensor.shape[-2] // 2
                end_x, end_y = start_x + self.wd_tensor.shape[-1], start_y + self.wd_tensor.shape[-2]
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
                if end_y > img_tensor.shape[2]:
                    end_y = img_tensor.shape[2]
                    start_y = end_y - self.wd_tensor.shape[-1]
                if end_x > img_tensor.shape[3]:
                    end_x = img_tensor.shape[3]
                    start_x = end_x - self.wd_tensor.shape[-2]
                if start_y < 0:
                    start_y = 0
                    end_y = self.wd_tensor.shape[-1]
                if start_x < 0:
                    start_x = 0
                    end_x = self.wd_tensor.shape[-2]
                img_tensor[bi, :, start_y:end_y, start_x:end_x] = self.wd_tensor
        return img_tensor

    def save_wd(self, epoch):
        wd_img = transforms.ToPILImage('RGB')(self.wd_tensor)
        wd_img = np.array(wd_img).astype(np.uint8)
        wd_img = cv2.cvtColor(wd_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(f'{self.wd_save_path}', f'wd_{str(epoch).zfill(3)}.png'), wd_img)
        del wd_img
        pass

    def eval_single(self, img_cv):
        self.detector.detector.eval()
        img_original_result, possible_person_ls = self.detector.detect_for_hidden_person(img_cv, interest_score_thresh=self.person_conf)
        if len(possible_person_ls) == 0:
            return False
        else:
            if len(img_original_result) and len(img_original_result[0]) > 0:
                detected_person_in_original_ls = img_original_result[np.where(img_original_result[:, -1] == 1)]
            else:
                detected_person_in_original_ls = []
            img_sized_with_wd = self.eval_add_wd_into_img(img_cv, possible_person_ls)
            img_with_wd_result = self.detector.detect_single_img_cv(img_sized_with_wd)
            if len(img_with_wd_result) and len(img_with_wd_result[0]) > 0:
                detected_person_in_wd_ls = img_with_wd_result[np.where(img_with_wd_result[:, -1] == 1)]
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
                    if max(temp_ls) < 0.5:
                        has_new_person = True
                        break
            return has_new_person

    def eval_add_wd_into_img(self, img_sized_with_wd, possible_person_ls):
        add_eval_wd = copy.deepcopy(self.eval_wd)
        for single_person_xyxy in possible_person_ls:
            start_x = int(single_person_xyxy[0])
            start_y = int(single_person_xyxy[1])
            end_x = int(single_person_xyxy[2])
            end_y = int(single_person_xyxy[3])
            cx = (start_x + end_x) // 2
            cy = (start_y + end_y) // 2
            start_x, start_y = cx - add_eval_wd.shape[1] // 2, cy - add_eval_wd.shape[0] // 2
            end_x, end_y = start_x + add_eval_wd.shape[1], start_y + add_eval_wd.shape[0]
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
            if end_y > img_sized_with_wd.shape[0]:
                end_y = img_sized_with_wd.shape[0]
                start_y = end_y - add_eval_wd.shape[0]
            if end_x > img_sized_with_wd.shape[1]:
                end_x = img_sized_with_wd.shape[1]
                start_x = end_x - add_eval_wd.shape[1]
            if start_y < 0:
                start_y = 0
                end_y = add_eval_wd.shape[0]
            if start_x < 0:
                start_x = 0
                end_x = add_eval_wd.shape[1]
            img_sized_with_wd[start_y:end_y, start_x:end_x, :] = add_eval_wd
        return img_sized_with_wd

    def eval_load_wd(self, wd_path):
        wd_img = cv2.imread(wd_path, 1)
        if (wd_img.shape[0] != self.cfg.wd_size) or (wd_img.shape[1] != self.cfg.wd_size):
            wd_img = cv2.resize(wd_img, (self.cfg.wd_size, self.cfg.wd_size))
        self.eval_wd = wd_img
        print(f'Load woodpecker for eval: {wd_path}')
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
        train_dataset = WoodpeckerFasterRCNNDataset(self.cfg.clean_root, self.cfg.attack_root, self.cfg.clean_hidden_root, self.cfg.attack_hidden_root, self.cfg.attack_label_root, self.cfg.wd_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_works)
        self.data_loader = train_loader
        pass

    def init_woodpecker(self):
        wd_img = np.random.randint(0, 255, (3, self.cfg.wd_size, self.cfg.wd_size)) / 255.
        wd_tensor = torch.from_numpy(wd_img)
        self.wd_tensor = wd_tensor.requires_grad_(True)
        self.wd_save_path = os.path.join('./FJNTraining', 
                                         f'wd_{self.cfg.attack_name}_{self.cfg.detect_name}_{self.cfg.weight}_{self.cfg.batch_size}',
                                         f'exp_{self.cfg.wd_size}')
        os.makedirs(self.wd_save_path, exist_ok=True)
        del wd_img
        self.save_wd(0)
        pass



def get_args():
    Gparser = argparse.ArgumentParser(description='faster r-cnn cawd')

    Gparser.add_argument('--detect_name', default='fsrcnn', type=str, help='')
    Gparser.add_argument('--attack_name', default='advtext', type=str, help='')
    Gparser.add_argument('--weight', default=1.0, type=float, help='weight of benign loss')
    Gparser.add_argument('--defensive_patch_location', default='cc', type=str, help='defensive patch location', choices=['uc', 'cl', 'cr', 'bc', 'cc'])
    Gparser.add_argument('--canary_init_path', default='InitImages/FSRCNN/', type=str, help='canary init image')
    Gparser.add_argument('--canary_init', action='store_true', default=True, help='options :True or False')
    Gparser.add_argument('--canary_cls_id', default=24, type=int, help='canary label')
    Gparser.add_argument('--canary_size', default=120, type=int, help='canary size')
    Gparser.add_argument('--wd_size', default=120, type=int, help='canary size')
    Gparser.add_argument('--data_name', default="VOC16", type=str)
    
    Gparser.add_argument('--clean_root', default="Data/traineval/VOC07_FSRCNN/train16/benign", type=str)
    Gparser.add_argument('--clean_hidden_root', default="Data/traineval/VOC07_FSRCNN/train16/benign_hidden", type=str)
    Gparser.add_argument('--attack_root', default="Data/traineval/VOC07_FSRCNN/train16/adversarial", type=str)
    Gparser.add_argument('--attack_hidden_root', default="Data/traineval/VOC07_FSRCNN/train16/adversarial_hidden", type=str)
    Gparser.add_argument('--attack_label_root', default="Data/traineval/VOC07_FSRCNN/train16/adversarial_label", type=str)
    
    Gparser.add_argument('--epoch', default=50, type=int, help='epoch for train')
    Gparser.add_argument('--learing_rate', default=0.05, type=float, help='batch_size for training')
    Gparser.add_argument('--epoch_save', default=1, type=int, help='epoch for save model and canary')
    Gparser.add_argument('--person_conf', default=0.075, type=float, help='person_conf')
    Gparser.add_argument('--overlap_thresh', default=0.4, type=float, help='overlap_thresh')
    Gparser.add_argument('--margin_size', default=40, type=int, help='margin size')
    Gparser.add_argument('--seed', default=301, type=int, help='choose seed')
    Gparser.add_argument('--shuffle', action='store_true', default=True, help='options :True or False')
    Gparser.add_argument('--batch_size', default=1, type=int, help='batch_size for training')
    Gparser.add_argument('--num_works', default=0, type=int, help='num_works')
    Gparser.add_argument('--train', action='store_true', help='train')
    Gparser.add_argument('--test', action='store_true', help='eval')
    Gparser.add_argument('--make_labs', action='store_true', help='make_labs')
    Gparser.add_argument('--gpu_id', default=0, type=int, help='choose gpu')
    Gparser.add_argument('--df_mode', default='C', type=str, help='select df_patch', choices=['C', 'W', 'A'])
    Gparser.add_argument('--input_img', default='', type=str, help='the patch of input img')
    Gparser.add_argument('--best_canary_path', default='./trained_dfpatches/FSRCNN/canary.png', type=str, help='the patch of best_canary')
    Gparser.add_argument('--best_wd_path', default='./trained_dfpatches/FSRCNN/wd.png', type=str, help='the patch of best_wd')
    
    return Gparser.parse_known_args()[0]

def freeze_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def make_labs(detector, img_root, txt_root):
    os.makedirs( txt_root, exist_ok=True )
    for img_name in os.listdir( img_root ):
        text_name = img_name.split('.')[0]+'.txt'
        img_cv = cv2.imread( os.path.join( img_root, img_name ), 1 )
        original_res, possible_res = detector.detect_for_hidden_person(img_cv)
        
        textfile = open(os.path.join( txt_root, text_name ), 'w+')
        for item in possible_res:
            # cx, cy = int((item[0] + item[2]) / 2) , int((item[1] + item[3]) / 2)
            # width, height = int(item[2] - item[0]) , int(item[3] - item[1]) 
            textfile.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(item[-1].item(), item[0].item(), item[1].item(), item[2].item(), item[3].item()))
        
        textfile.close()
        pass
    pass


if __name__ == '__main__':
    cfg = get_args()
    freeze_seed(cfg.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.gpu_id}'
    detector = fjn_faster_rcnn()
    
    if cfg.make_labs:
        make_labs(detector, cfg.clean_root, cfg.clean_hidden_root)
        make_labs(detector, cfg.attack_root, cfg.attack_hidden_root)
        
        os.makedirs(cfg.attack_label_root, exist_ok=True)
        for img_file in os.listdir(cfg.clean_root):
            if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                name = os.path.splitext(img_file)[0]
                input_imgs = cv2.imread(os.path.join(cfg.clean_root, img_file), 1)
                boxes = detector.detect_single_img_cv(input_imgs)
                textfile = open(os.path.join(cfg.attack_label_root, name + '.txt'), 'w+')
                for item in boxes:
                    cx, cy = (item[0] + item[2]) / 2, (item[1] + item[3]) / 2
                    width, height = (item[2] - item[0]), (item[3] - item[1])
                    textfile.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(item[-1].item(), cx, cy, width, height))
                textfile.close()
            pass
        pass
    
    
    if cfg.train:
        if cfg.df_mode == 'C':
            canary = Canary(cfg, detector)
            canary.init_dataloader()
            canary.init_canary()
            canary.train()
            pass
        elif cfg.df_mode == 'W':
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
        
        pass

    pass




'''

python FSRCNN_Combiner_New1209.py --make_labs --clean_root Data/traineval/VOC07_FSRCNN/train16/benign --clean_hidden_root Data/traineval/VOC07_FSRCNN/train16/benign_hidden --attack_root Data/traineval/VOC07_FSRCNN/train16/adversarial --attack_hidden_root Data/traineval/VOC07_FSRCNN/train16/adversarial_hidden --attack_label_root Data/traineval/VOC07_FSRCNN/train16/adversarial_label

python FSRCNN_Combiner_New1209.py --train --df_mode C --defensive_patch_location cc --canary_cls_id 24 --canary_size 120 --person_conf 0.075 --weight 2.0

python FSRCNN_Combiner_New1209.py --train --df_mode W --defensive_patch_location cc --wd_size 120 --person_conf 0.075 --weight 1.0

'''
