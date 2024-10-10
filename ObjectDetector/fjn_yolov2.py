import copy
import cv2
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from ObjectDetector.yolov2.utils import *
from ObjectDetector.yolov2.darknet import Darknet

import warnings
warnings.filterwarnings("ignore")


def person_and_original_region_boxes(output, conf_thresh, person_conf_thresh, num_classes, anchors, num_anchors):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    original_obj_boxs = []  
    all_person_boxs = []  

    output = output.view(batch * num_anchors, 5 + num_classes, h * w)
    output = output.transpose(0, 1).contiguous()
    output = output.view(5 + num_classes, batch * num_anchors * h * w)
    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)

    cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_ids = cls_max_ids.view(-1)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)

    for b in range(batch):

        original_obj_boxes = []  
        all_person_boxes = []  

        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx

                    det_conf = det_confs[ind]

                    if det_conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        original_obj_boxes.append(box)

                    if (cls_max_ids[ind] == 0) and (det_conf > person_conf_thresh):  
                        person_bcx = xs[ind]
                        person_bcy = ys[ind]
                        person_bw = ws[ind]
                        person_bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        person_box = [person_bcx / w, person_bcy / h, person_bw / w, person_bh / h, det_conf, cls_max_conf, cls_max_id]
                        all_person_boxes.append(person_box)

        original_obj_boxs.append(original_obj_boxes)
        all_person_boxs.append(all_person_boxes)

    return original_obj_boxs, all_person_boxs


def person_and_original_region_boxes_faster(output, conf_thresh, person_conf_thresh, num_classes, anchors, num_anchors, print_time=False):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    original_obj_boxs = []  
    all_person_boxs = []  

    output = output.view(batch * num_anchors, 5 + num_classes, h * w)


    output = output[1:,::]
    anchors = anchors[2:]
    num_anchors = num_anchors-1

    anchor_step = len(anchors) // num_anchors

    output = output.transpose(0, 1).contiguous()
    output = output.view(5 + num_classes, batch * num_anchors * h * w)
    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w).cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)

    t2 = time.time()

    for b in range(batch):

        original_obj_boxes = []  
        all_person_boxes = []  

        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx

                    det_conf = det_confs[ind]

                    if det_conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        original_obj_boxes.append(box)

                    if (cls_max_ids[ind] == 0) and (det_conf > person_conf_thresh):  
                        person_bcx = xs[ind]
                        person_bcy = ys[ind]
                        person_bw = ws[ind]
                        person_bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        person_box = [person_bcx / w, person_bcy / h, person_bw / w, person_bh / h, det_conf, cls_max_conf, cls_max_id]
                        all_person_boxes.append(person_box)

        original_obj_boxs.append(original_obj_boxes)
        all_person_boxs.append(all_person_boxes)
    t3 = time.time()
    if print_time:
        print('---------------------------------')
        print('matrix computation : %f ms' % ((t1 - t0)*1000))
        print('        gpu to cpu : %f ms' % ((t2 - t1)*1000))
        print('      boxes filter : %f ms' % ((t3 - t2)*1000))
        print('---------------------------------')
    return original_obj_boxs, all_person_boxs




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

                        if cal_overlap(single_res[:4], group_single_res[:4], x1y1x2y2=True) >=0.4:
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

    return max(carea / area1, carea / area2)

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

    return carea



def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh_np(x):
    y = np.zeros_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


class FJN_YOLOV2():
    def __init__(self, num_classes=80):
        self.num_classes = num_classes
        self.class_names = []
        self.model = None

        self.set_init()
        pass

    def get_model_output_from_img_cv(self, img_cv):
        img_sized = copy.deepcopy(img_cv)
        if img_sized.shape[0] != self.model.height or img_sized.shape[1] != self.model.width:
            img_sized = cv2.resize(img_sized, (self.model.width, self.model.height))
        img_sized = cv2.cvtColor(img_sized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        img_tensor = torch.autograd.Variable(img_tensor)
        output = self.model.forward(img_tensor)  
        return output

    def detect_single(self, img):
        if isinstance(img, np.ndarray):
            output = self.get_model_output_from_img_cv(img)
        elif isinstance(img, torch.Tensor):
            output = self.model.forward(img)
        else:
            output = None

        boxes = get_region_boxes(output, self.conf_thresh, self.model.num_classes, self.model.anchors, self.model.num_anchors)[0]

        boxes = nms(boxes, self.nms_thresh)

        new_boxes = []
        for idx in range(len(boxes)):
            tmp_res = [res.detach().item() for res in boxes[idx]]
            for loc_i in [0,1,2,3]:
                tmp_res[loc_i] = max(min(tmp_res[loc_i], 1.0), 0.0)
            new_boxes.append(tmp_res)
        new_boxes = np.array([[0, 0, 0, 0, 0, 0, -1]]) if len(new_boxes) < 1 else np.array(new_boxes)
        return new_boxes

    def detect_person_and_original(self, img, person_conf=0.05, faster=False):
        if isinstance(img, np.ndarray):
            output = self.get_model_output_from_img_cv(img)
        elif isinstance(img, torch.Tensor):
            output = self.model.forward(img)
        if faster:
            original_obj_boxs, all_person_boxs = person_and_original_region_boxes_faster(output, self.conf_thresh, person_conf, self.model.num_classes, self.model.anchors, self.model.num_anchors)
        else:
            original_obj_boxs, all_person_boxs = person_and_original_region_boxes(output, self.conf_thresh, person_conf, self.model.num_classes, self.model.anchors, self.model.num_anchors)


        original_obj_boxs = original_obj_boxs[0]
        original_obj_boxs = nms(original_obj_boxs, self.nms_thresh)
        all_person_boxs = all_person_boxs[0]

        return_person_boxes = []
        for idx in range(len(all_person_boxs)):
            return_person_boxes.append([(res.detach().item()) for res in all_person_boxs[idx]])
        return_person_boxes = np.array([[0, 0, 0, 0, 0, 0, -1]]) if len(return_person_boxes) < 1 else np.array(return_person_boxes)

        return_original_boxes = []
        for idx in range(len(original_obj_boxs)):
            return_original_boxes.append([res.detach().item() for res in original_obj_boxs[idx]])
        return_original_boxes = np.array([[0, 0, 0, 0, 0, 0, -1]]) if len(return_original_boxes) < 1 else np.array(return_original_boxes)



        return return_person_boxes, return_original_boxes

    def draw_detect_single_image(self, img_cv):
        image = copy.deepcopy(img_cv)
        boxes = self.detect_single(image)
        width = image.shape[1]
        height = image.shape[0]
        for x, y, w, h, conf, cls_conf, cls_pred in boxes:
            if int(cls_pred) == -1:
                continue

            if cls_pred == 0:
                color = self.bbox_colors[int(cls_pred)]
                color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            else:
                color = self.bbox_colors[int(cls_pred)]
                color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

            x1 = int(round((x - w / 2.0) * width))
            y1 = int(round((y - h / 2.0) * height))
            x2 = int(round((x + w / 2.0) * width))
            y2 = int(round((y + h / 2.0) * height))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)

            cv2.rectangle(image, (x1 - 2, y1), (x2 + 2, y1 + 19), color, thickness=-1)
            if cls_pred == 0:
                cv2.putText(image, self.class_names[int(cls_pred)] + " {:.2f}".format(conf), (x1, y1 + 13), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)
            else:
                
                cv2.putText(image, self.class_names[int(cls_pred)] + " {:.2f}".format(conf), (x1, y1 + 13), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)



        return image

    def draw_single_image_with_box(self, img_cv, boxes):
        image = copy.deepcopy(img_cv)
        width = image.shape[1]
        height = image.shape[0]
        for x, y, w, h, conf, cls_conf, cls_pred in boxes:
            if int(cls_pred) == -1:
                continue




            color = self.bbox_colors[int(cls_pred)]
            color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

                
            x1 = int(round((x - w / 2.0) * width))
            y1 = int(round((y - h / 2.0) * height))
            x2 = int(round((x + w / 2.0) * width))
            y2 = int(round((y + h / 2.0) * height))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)

            cv2.rectangle(image, (x1 - 2, y1), (x2 + 2, y1 + 19), color, thickness=-1)
            

            cv2.putText(image, self.class_names[int(cls_pred)] + " {:.2f}".format(conf), (x1, y1 + 13), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

        return image

    def FindHiddenPerson(self, img, person_conf=0.05, overlap_thresh=0.4, faster=False, shown=False):

        box_all_person, box_original_obj = self.detect_person_and_original(img, person_conf=person_conf, faster=faster)
        box_original_person = box_original_obj[np.where(box_original_obj[:, -1] == 0)]

        if shown:
            cv2.imshow('all_person_draw', self.draw_single_image_with_box(img, box_all_person))

        box_all_person_overlap = []
        if len(box_original_person) > 0:

            for single_box_all_person in box_all_person:
                boo_is_overlap = False

                for single_box_original_person in box_original_person:

                    bool_x_in_left = (single_box_original_person[0] - single_box_original_person[2]/2) < single_box_all_person[0]
                    bool_x_in_right= single_box_all_person[0] < (single_box_original_person[0] + single_box_original_person[2]/2)
                    bool_y_in_top = (single_box_original_person[1] - single_box_original_person[3]/2) < single_box_all_person[1]
                    bool_y_in_bottom= single_box_all_person[1] < (single_box_original_person[1] + single_box_original_person[3]/2)
                    if bool_x_in_left and bool_x_in_right and bool_y_in_top and bool_y_in_bottom:
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

                if tmp_overlap_area/(0.098*0.098) > 0.3:
                    bool_tmp_is_defense_overlap = True

            if not bool_tmp_is_defense_overlap:
                new_box_all_person_overlap.append(single_box_person_overlap)
                pass
            pass

        if shown:
            cv2.imshow('new_box_all_person_overlap', self.draw_single_image_with_box(img, new_box_all_person_overlap))

        if len(new_box_all_person_overlap) > 0:
            new_all_person_boxes = np.array(new_box_all_person_overlap)
            new_all_person_xyxy_boxes = xywh2xyxy_np(new_all_person_boxes[:, :4])
            new_all_person_xyxy_boxes = np.minimum(np.maximum(new_all_person_xyxy_boxes, 0), 1.0)
            person_groups_xyxy = divid_person_group(new_all_person_xyxy_boxes)

            hidden_xywh = xyxy2xywh_np(person_groups_xyxy)
            hidden_box = np.zeros((len(hidden_xywh), 7))
            hidden_box[:, :4] = hidden_xywh
            hidden_xywh = hidden_box
        else:
            hidden_xywh = np.array([])


        return hidden_xywh

    def FindHiddenPerson_folds(self, img_root, save_root, person_conf=0.05, overlap_thresh=0.4, faster=False):
        os.makedirs(save_root, exist_ok=True)

        img_name_ls = os.listdir(img_root)
        t = tqdm(total=len(img_name_ls), ascii=True)
        t.set_description(f'FindHiddenPerson_folds on {img_root}')
        for img_name in img_name_ls:
            img_path = os.path.join(img_root, img_name)
            img_cv = cv2.imread(img_path, 1)

            hidden_boxes =self. FindHiddenPerson(img_cv, person_conf=person_conf, overlap_thresh=overlap_thresh, faster=faster)
            hidden_person_box_detect = self.draw_single_image_with_box(img_cv, hidden_boxes)

            cv2.imwrite(os.path.join(save_root, img_name), hidden_person_box_detect)
            t.update(1)
        t.close()
        pass

    def draw_possible_person(self, img_cv, faster=False):
        hidden_boxes = self.FindHiddenPerson(img_cv, faster=faster)
        image = copy.deepcopy(img_cv)
        width = image.shape[1]
        height = image.shape[0]
        for x, y, w, h, conf, cls_conf, cls_pred in hidden_boxes:
            if int(cls_pred) == -1:
                continue

            color = (255, 255, 255)

            x1 = int(round((x - w / 2.0) * width))
            y1 = int(round((y - h / 2.0) * height))
            x2 = int(round((x + w / 2.0) * width))
            y2 = int(round((y + h / 2.0) * height))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)


        return image

    def draw_all_person_folds(self, img_root, save_root, person_conf=0.05, faster=False):
        os.makedirs(save_root, exist_ok=True)

        img_name_ls = os.listdir(img_root)
        t = tqdm(total=len(img_name_ls), ascii=True)
        t.set_description(f'Draw all person boxes folds on {img_root}')
        for img_name in img_name_ls:
            img_path = os.path.join(img_root, img_name)
            img_cv = cv2.imread(img_path, 1)

            box_all_person, box_original_obj = self.detect_person_and_original(img_cv, person_conf=person_conf, faster=faster)
            cv2.imwrite(os.path.join(save_root, img_name), self.draw_single_image_with_box(img_cv, box_all_person))
            t.update(1)
        t.close()
        pass

    def detect_grid_cell(self, img_cv):
        detect_output = self.get_model_output_from_img_cv(img_cv)
        img_sized = cv2.resize(img_cv, (self.model.width * 4, self.model.height * 4))
        split_num = 19
        splid_len = 32 * 4
        for i in range(split_num - 1):
            cv2.line(img_sized, (splid_len * (i + 1), 0), (splid_len * (i + 1), 608 * 4), (0, 255, 0), 1, 4)  
            cv2.line(img_sized, (0, splid_len * (i + 1)), (608 * 4, splid_len * (i + 1)), (0, 255, 0), 1, 4)  

        num_anchors = self.model.num_anchors
        num_classes = self.model.num_classes

        if detect_output.dim() == 3:
            detect_output = detect_output.unsqueeze(0)
        batch = detect_output.size(0)  
        assert (detect_output.size(1) == (5 + num_classes) * num_anchors)
        h = detect_output.size(2) 
        w = detect_output.size(3) 

        output = detect_output.view(batch * num_anchors, 5 + num_classes, h * w)  
        output = output.transpose(0, 1).contiguous()  
        output = output.view(5 + num_classes, batch * num_anchors * h * w)  

        det_confs = torch.sigmoid(output[4])
        cls_confs = torch.nn.Softmax()(Variable(output[5:5 + num_classes].transpose(0, 1))).data
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_ids = cls_max_ids.view(-1)

        sz_hw = h * w
        sz_hwa = sz_hw * num_anchors
        det_confs = convert2cpu(det_confs)
        cls_max_ids = convert2cpu_long(cls_max_ids)
        for b in range(batch):
            for cy in range(h):
                for cx in range(w):
                    for i in range(num_anchors):
                        ind = b * sz_hwa + i * sz_hw + cy * w + cx

                        if det_confs[ind].item() > 0.5:
                            cv2.putText(img_sized,
                                        "{:.2f} {}".format(det_confs[ind].item(), self.class_names[cls_max_ids[ind].item()]),
                                        (8 + cx * splid_len, 16 + cy * splid_len + i * 16), cv2.FONT_HERSHEY_COMPLEX,
                                        0.6, (0, 0, 255), 1, 4)
                        else:
                            cv2.putText(img_sized,
                                        "{:.2f} {}".format(det_confs[ind].item(), self.class_names[cls_max_ids[ind].item()]),
                                        (8 + cx * splid_len, 16 + cy * splid_len + i * 16), cv2.FONT_HERSHEY_COMPLEX,
                                        0.6, (0, 255, 0), 1, 4)

        return img_sized

    def get_detect_mask(self, img_cv, person_only=False):
        boxes = self.get_model_output_from_img_cv(img_cv)
        original_detect_mask = np.zeros((img_cv.shape[0], img_cv.shape[1], 3))
        for single_detect in boxes:
            if person_only:
                if int(single_detect[-1]) != 0:
                    continue
            single_center_x = int(single_detect[0] * img_cv.shape[1])
            single_center_y = int(single_detect[1] * img_cv.shape[0])
            single_width = int(single_detect[2] * img_cv.shape[1])
            single_height = int(single_detect[3] * img_cv.shape[0])
            single_start_x = max(single_center_x - single_width // 2, 0)
            single_start_y = max(single_center_y - single_height // 2, 0)
            single_end_x = min(single_center_x + single_width // 2, img_cv.shape[1])
            single_end_y = min(single_center_y + single_height // 2, img_cv.shape[0])

            original_detect_mask[single_start_y:single_end_y, single_start_x:single_end_x, :] = 1
        return original_detect_mask

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

    def get_label(self, imgs_path, save_txt_path):
        os.makedirs(save_txt_path, exist_ok=True)

        for img_file in os.listdir(imgs_path):
            if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                name = os.path.splitext(img_file)[0]  
                input_imgs = cv2.imread(os.path.join(imgs_path, img_file), 1)
                boxes = self.detect_single(input_imgs)

                textfile = open(os.path.join(save_txt_path, name + '.txt'), 'w+')

                for item in boxes:
                    x, y = item[0].item(), item[1].item()
                    width, height = item[2].item(), item[3].item()
                    textfile.write('{:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(item[-1].item(), x, y, width, height))

                textfile.close()

    def get_patch_area(self, imgs_path, save_txt_path):
        os.makedirs(save_txt_path, exist_ok=True)
        for img_file in os.listdir(imgs_path):
            if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                name = os.path.splitext(img_file)[0]
                input_imgs = cv2.imread(os.path.join(imgs_path, img_file), 1)

                all_box = self.detect_single(input_imgs)
                person_box = all_box[np.where(all_box[:, -1] == 0)]
                original_person_num = len(person_box)

                if original_person_num == 1:
                    biggest_person = person_box
                elif original_person_num > 1:
                    most_area_ls = sort_by_area(person_box)
                    biggest_person = [person_box[int(most_area_ls[0])]]

                img_size = input_imgs.shape
                center_x, center_y, w, h = biggest_person[0][:4]

                center_x, w = int(center_x * img_size[1]), int(w * img_size[1])
                center_y, h = int(center_y * img_size[0]), int(h * img_size[0])

                length = int(min(w, h) * 0.9)
                length = min(length, int(min(img_size[0], img_size[1]) * 0.2))

                center_x, center_y, length_w, length_h = center_x / img_size[1], center_y / img_size[0], length / img_size[1], length / img_size[0]


                textfile = open(os.path.join(save_txt_path, name + '.txt'), 'w+')
                textfile.write('{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(center_x, center_y, length_w, length_h))
                textfile.close()


    def get_patch_area_multi_patches(self, imgs_path, save_txt_path):
        os.makedirs(save_txt_path, exist_ok=True)
        for img_file in os.listdir(imgs_path):
            if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                name = os.path.splitext(img_file)[0] 

                input_imgs = cv2.imread(os.path.join(imgs_path, img_file), 1)

                all_box = self.detect_single(input_imgs)
                person_box = all_box[np.where(all_box[:, -1] == 0)]
                most_area_ls = sort_by_area(person_box)

                textfile = open(os.path.join(save_txt_path, name + '.txt'), 'w+')

                for big_no in most_area_ls:
                    single_person = person_box[int(big_no)]

                    img_size = input_imgs.shape
                    center_x, center_y, w, h = single_person[:4]

                    center_x, w = int(center_x * img_size[1]), int(w * img_size[1])
                    center_y, h = int(center_y * img_size[0]), int(h * img_size[0])

                    length = int(min(w, h) * 0.9)
                    length = min(length, int(min(img_size[0], img_size[1]) * 0.2))

                    center_x, center_y, length_w, length_h = center_x / img_size[1], center_y / img_size[0], length / img_size[1], length / img_size[0]

                    if length_w<20/608 or length_h < 20/608:
                        continue


                    textfile.write('{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(center_x, center_y, length_w, length_h))

                textfile.close()

    def set_init(self):
        self.init_names()
        self.init_model()
        self.init_draw()

        pass

    def init_names(self):
        if self.num_classes == 20:
            namesfile = 'ObjectDetector/yolov2/voc.names'
        elif self.num_classes == 80:
            namesfile = 'ObjectDetector/yolov2/coco.names'
        else:
            namesfile = 'ObjectDetector/yolov2/names'

        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            self.class_names.append(line)

        pass

    def init_model(self):
        weightfile = 'ObjectDetector/yolov2/yolov2.weights'
        cfgfile = 'ObjectDetector/yolov2/yolov2.cfg'
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightfile)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        self.conf_thresh = 0.5
        self.nms_thresh = 0.4
        pass

    def init_draw(self):
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 80)]
        random.seed(1)
        self.bbox_colors = random.sample(colors, 80)
        pass












