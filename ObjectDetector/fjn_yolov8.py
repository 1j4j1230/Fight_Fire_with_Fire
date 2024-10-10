import torch
import platform
from copy import deepcopy
import torchvision
import time
import cv2
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.utils.torch_utils import select_device
from ultralytics.models import yolo
from ultralytics.utils import ops
from ultralytics.utils.ops import xywh2xyxy, box_iou
import copy
import random
from tqdm import tqdm
import os


from ultralytics.nn.autobackend import AutoBackend


def non_max_suppression_hidden(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        hidden_conf=0.05,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    
    
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)): 
        prediction = prediction[0]  

    device = prediction.device


    bs = prediction.shape[0] 
    nc = nc or (prediction.shape[1] - 4)  
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  



    xc = prediction[:, 4:mi].amax(1) > conf_thres 

    xc_hidden = prediction[:, 4] > hidden_conf  

   
    time_limit = 0.5 + max_time_img * bs  
    redundant = True  
    multi_label &= nc > 1  
    merge = False  

    prediction = prediction.transpose(-1, -2) 


    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    output_hidden = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):  
        
        x_hidden = x[xc_hidden[xi]] 
        x = x[xc[xi]]  

     
        if not x_hidden.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)
        box_hidden, cls_hidden, mask_hidden = x_hidden.split((4, nc, nm), 1) 


        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else: 
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            conf_hidden, j_hidden = cls_hidden.max(1, keepdim=True) 
            x_hidden = torch.cat((box_hidden, conf_hidden, j_hidden.float(), mask_hidden), 1)[j_hidden.view(-1) == 0.] 


        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  
        n_hidden = x_hidden.shape[0]  

        if (not n) and (not n_hidden): 
            continue


        if n > max_nms:  
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  
        if n_hidden > max_nms: 
            x_hidden = x_hidden[x_hidden[:, 4].argsort(descending=True)[:max_nms]] 


        c = x[:, 5:6] * (0 if agnostic else max_wh)  
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres) 
        i = i[:max_det] 


        c_hidden = x_hidden[:, 5:6] * (0 if agnostic else max_wh) 
        boxes_hidden, scores_hidden = x_hidden[:, :4] + c_hidden, x_hidden[:, 4]  
        i_hidden = torchvision.ops.nms(boxes_hidden, scores_hidden, iou_thres)  
        i_hidden = i_hidden[:max_det] 


        if merge and (1 < n < 3E3): 
          
            iou = box_iou(boxes[i], boxes) > iou_thres  
            weights = iou * scores[None]  
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True) 
            if redundant:
                i = i[iou.sum(1) > 1] 

        output[xi] = x[i]
        output_hidden[xi] = x_hidden[i_hidden]



        if (time.time() - t) > time_limit:
            
            break  

    return output, output_hidden



def plot_one_box(x, img, color=None, label=None, line_thickness=None):

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1 
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1) 
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  
    y[:, 2] = x[:, 2] - x[:, 0]  
    y[:, 3] = x[:, 3] - x[:, 1]  
    return y



def cal_overlap(box1, box2, x1y1x2y2=False):
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

    return carea / area1


def sort_by_area(detect_person_res):
    area_ls = []
    for i, single_res in enumerate(detect_person_res):
        temp_area = []
        s = (single_res[3] - single_res[1]) * (single_res[2] - single_res[0])
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

                        tmp_overlap_res = cal_overlap(single_res[:4], group_single_res[:4], x1y1x2y2=True)
                        if tmp_overlap_res >=0.04:
                            res_is_overlap = True
                            overlap_group[group_id][:2] = np.minimum(group_single_res[:2], single_res[:2])
                            overlap_group[group_id][2:4] = np.maximum(group_single_res[2:4], single_res[2:4])
                            break

                if not res_is_overlap:
                    overlap_group.append(single_res)
        return overlap_group

    first_overlap_group = combine_over_lap(deal_person_detect_res)

    second_overlap_group = combine_over_lap(first_overlap_group)

    while len(first_overlap_group) != len(second_overlap_group):
        first_overlap_group = combine_over_lap(second_overlap_group)
        second_overlap_group = combine_over_lap(first_overlap_group)
    return np.array(second_overlap_group)




def box_in_another_box(single_box_original_person, single_box_all_person):

    bool_x_in_left = (single_box_all_person[0] + single_box_all_person[2] )/ 2 > single_box_original_person[0]
    if not bool_x_in_left:
        return False

    bool_x_in_right = (single_box_all_person[0] + single_box_all_person[2] )/ 2 < single_box_original_person[2]
    if not bool_x_in_right:
        return False

    bool_y_in_top = (single_box_all_person[1] + single_box_all_person[3] )/ 2 > single_box_original_person[1]
    if not bool_y_in_top:
        return False

    bool_y_in_bottom = (single_box_all_person[1] + single_box_all_person[3] )/ 2 < single_box_original_person[3]
    if not bool_y_in_bottom:
        return False

    return True



def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):  
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])  
        boxes[..., 2].clamp_(0, shape[1])  
        boxes[..., 3].clamp_(0, shape[0])  
    else:  
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  


def scale_boxes(img1_shape, boxes, img0_shape):

    gain_x = img1_shape[1] / img0_shape[1]
    gain_y = img1_shape[0] / img0_shape[0]
    
    boxes[..., 0] = boxes[..., 0] / gain_x
    boxes[..., 2] = boxes[..., 2] / gain_x
    boxes[..., 1] = boxes[..., 1] / gain_y
    boxes[..., 3] = boxes[..., 3] / gain_y

    
    clip_boxes(boxes, img0_shape)
    return boxes




class FJN_YOLOv8:

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.imgsz = 640
        self.conf = 0.25
        self.iou = 0.7
        self.agnostic_nms = False
        self.max_det = 300
        self.classes = None

        self.model = None
        self.init_model(self.model)

        random.seed(1)
        self.names = self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        self.done_warmup = False
        pass

    def pre_transform(self, im):
        return [cv2.resize(deepcopy(x), (self.imgsz, self.imgsz)) if (x.shape[0] != self.imgsz or x.shape[1] != self.imgsz) else x   for x in im]
    

    def pre_proposse_img_cv(self, im):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform([im]))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  
            im = np.ascontiguousarray(im)  
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  
        if not_tensor:
            img /= 255  
        return img

    def detect_single(self, img_original):
        if not self.done_warmup:
            self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
            self.done_warmup = True

        img_original_is_np = isinstance(img_original, np.ndarray)
        if img_original_is_np:
            orig_img_shape = img_original.shape
            img = self.pre_proposse_img_cv(img_original)
        else:
            orig_img_shape = (img_original.shape[2], img_original.shape[3], 3)
            img = img_original

        with torch.no_grad():
            preds = self.model(img, augment=False, visualize=False)
            preds = ops.non_max_suppression(preds,
                                            self.conf,
                                            self.iou,
                                            agnostic=self.agnostic_nms,
                                            max_det=self.max_det,
                                            classes=self.classes)

            results = []
            for i, pred in enumerate(preds):
                if img_original_is_np:
                    pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img_shape)
                results.append(pred.cpu().numpy())
            return results

    
    def detect_fold_images(self, imgs_path, save_path):
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

    
    def draw_detect_single_image(self, img_cv):
        img_cv = copy.deepcopy(img_cv)
        det = self.detect_single(img_cv)[0]
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box(xyxy, img_cv, label=label, color=self.colors[int(cls)], line_thickness=3)
        return img_cv

    
    def draw_single_image_with_box(self, img_cv, box):
        img_cv = copy.deepcopy(img_cv)
        if box is not None and len(box[0]):
            for *xyxy, conf, cls in box[0]:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box(xyxy, img_cv, label=label, color=self.colors[int(cls)], line_thickness=3)
        return img_cv

    
    def get_labels(self, img_root, save_root):
        os.makedirs(save_root, exist_ok=True)

        img_name_ls = os.listdir(img_root)
        t = tqdm(total=len(img_name_ls), ascii=True)
        for img_name in img_name_ls:
            if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png'):
                txt_name = img_name.replace('.' + img_name.split('.')[-1], '.txt')
                txt_path = os.path.join(save_root, txt_name)
                img_cv = cv2.imread(os.path.join(img_root, img_name), 1)

                gn = torch.tensor(img_cv.shape)[[1, 0, 1, 0]] 

                det = self.detect_single(img_cv)[0]

                with open(txt_path, 'w+') as f:
                    for *xyxy, conf, cls in det:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh)) 
            t.update(1)
        t.close()
        pass


    def init_model(self, model=None, verbose=False):
        self.model = AutoBackend(model or 'ultralytics/pretrain/yolov8n.pt',
                                 device=select_device(self.device, verbose=verbose),
                                 dnn=False,
                                 data='ultralytics/cfg/datasets/coco.yaml',
                                 fp16=False,
                                 fuse=True,
                                 verbose=verbose)
        self.model.eval()
        pass



    def FindHiddenPerson(self, img, person_conf=0.05, overlap_thresh=0.4, remove_small_length=20, faster=False, shown=False):
        box_original_res, box_all_person = self.detect_person_and_original(img, person_conf=person_conf, faster=faster)

        box_original_person = [ single_orig_box[np.where(single_orig_box[:, -1] == 0)] for single_orig_box in box_original_res ]

        if shown:
            print('box_original_res: ', box_original_res)
            print('box_all_person: ', box_all_person)
            print('box_original_person: ', box_original_person)
            cv2.imshow('box_all_person', self.draw_single_image_with_box(img, box_all_person))
            cv2.imshow('box_original_res', self.draw_single_image_with_box(img, box_original_res))

        box_all_person_overlap = []
        if len(box_original_person[0]) > 0:
            for single_box_all_person in box_all_person[0]:
                boo_is_overlap = False
                for single_box_original_person in box_original_person[0]:

                    if box_in_another_box(single_box_original_person, single_box_all_person):
                        boo_is_overlap = True
                        break

                    if cal_overlap(single_box_all_person[:4], single_box_original_person[:4], x1y1x2y2=True) >= overlap_thresh:
                        boo_is_overlap = True
                        break
                if not boo_is_overlap:
                    box_all_person_overlap.append(single_box_all_person)

            box_all_person_overlap = [box_all_person_overlap]
        else:
            box_all_person_overlap = box_all_person

        if shown:
            print('box_all_person_overlap: ', box_all_person_overlap)
            cv2.imshow('box_all_person_overlap', self.draw_single_image_with_box(img, box_all_person_overlap))

        if len(box_all_person_overlap[0]) > 0:
            new_all_person_xyxy_boxes = np.array(box_all_person_overlap)
            person_groups_xyxy = divid_person_group(new_all_person_xyxy_boxes[0])
            hidden_xyxy = np.zeros((len(person_groups_xyxy), 6))
            hidden_xyxy[:, :4] = person_groups_xyxy
        else:
            hidden_xyxy = np.array([])

        final_hidden_xyxy = []
        for item in hidden_xyxy:
            if (item[2]-item[0])>remove_small_length and (item[3]-item[1])>remove_small_length:
                final_hidden_xyxy.append(item)
        hidden_xyxy = [np.array(final_hidden_xyxy)]

        if shown:
            print('hidden_xyxy: ', hidden_xyxy)
            cv2.imshow('hidden_xyxy', self.draw_single_image_with_box(img, hidden_xyxy))
            cv2.waitKey()
            cv2.destroyAllWindows()

        return hidden_xyxy


    def FindHidden_Fold(self, img_root, save_root, person_conf=0.05, overlap_thresh=0.4, remove_small_length=20, faster=False, combine=False):
        os.makedirs( save_root, exist_ok=True )
        img_name_ls = os.listdir(img_root)

        t = tqdm(total=len(img_name_ls), ascii=True)
        for img_name in img_name_ls:
            img_path = os.path.join(img_root, img_name)
            img_cv = cv2.imread(img_path, 1)
            img_cv = cv2.resize(img_cv, (640, 640))
            possible_area = self.FindHiddenPerson(img_cv, person_conf=person_conf, overlap_thresh=overlap_thresh, remove_small_length=remove_small_length, faster=faster)
            if combine:
                cv2.imwrite( os.path.join(save_root, img_name), np.hstack([self.draw_detect_single_image(img_cv), self.draw_single_image_with_box(img_cv, possible_area)]) )
            else:
                cv2.imwrite( os.path.join(save_root, img_name), self.draw_single_image_with_box(img_cv, possible_area) )
            t.set_postfix({f'detect image': img_name})
            t.update(1)
        t.close()
        pass


    def detect_person_and_original(self, img_original, person_conf=0.05, faster=False):
        with torch.no_grad():

            img_original_is_np = isinstance(img_original, np.ndarray)
            if img_original_is_np:
                orig_img_shape = img_original.shape
                img = self.pre_proposse_img_cv(img_original)
            else:
                orig_img_shape = (img_original.shape[2], img_original.shape[3], 3)
                img = img_original

            with torch.no_grad():
                preds = self.model(img, augment=False, visualize=False)


                preds, preds_hidden = non_max_suppression_hidden(preds,
                                                self.conf,
                                                self.iou,
                                                hidden_conf=person_conf,
                                                agnostic=self.agnostic_nms,
                                                max_det=self.max_det,
                                                classes=self.classes)
                results = []
                for i, pred in enumerate(preds):
                    if img_original_is_np:
                        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img_shape)
                    results.append(pred.cpu().numpy())
                results_hidden = []
                for i, pred_hidden in enumerate(preds_hidden):
                    if img_original_is_np:
                        pred_hidden[:, :4] = scale_boxes(img.shape[2:], pred_hidden[:, :4], orig_img_shape)
                    results_hidden.append(pred_hidden.cpu().numpy())

                return results, results_hidden




