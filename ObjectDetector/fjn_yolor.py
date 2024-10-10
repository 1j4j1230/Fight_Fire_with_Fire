import copy
import os
import torch
from tqdm import tqdm

from ObjectDetector.yolor.models.models import *
from ObjectDetector.yolor.utils.datasets import LoadImages, letterbox
from ObjectDetector.yolor.utils.plots import plot_one_box


def non_max_suppression_hidden(prediction, person_conf=0.05, conf_thres=0.1, iou_thres=0.6, classes=None, faster=False):
    if faster:
        re_prediction = None
        for item_idx, item in enumerate(prediction):
            item = item[:,1:,:,:,:]
            item_tmp = item.view(item.size(0), item.size(1)*item.size(2)*item.size(3), item.size(4))
            if item_idx == 0:
                re_prediction = item_tmp
            else:
                re_prediction = torch.cat([re_prediction, item_tmp], dim=1)
        prediction = re_prediction.contiguous()

    nc = prediction[0].shape[1] - 5 
    xc = prediction[..., 4] > conf_thres  
    xc_all = prediction[..., 4] > person_conf  


    min_wh, max_wh = 2, 4096 
    max_det = 300 
    multi_label = nc > 1  

    output = [torch.zeros(0, 6)] * prediction.shape[0]
    output_all = [torch.zeros(0, 6)] * prediction.shape[0] 

    for xi, x in enumerate(prediction):  
        x_all = x[xc_all[xi]]  
        x = x[xc[xi]]  

        x[:, 5:] *= x[:, 4:5]  
        x_all[:, 5:] *= x_all[:, 4:5] 

        box = xywh2xyxy(x[:, :4])
        box_all = xywh2xyxy(x_all[:, :4])   
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            i_all, j_all = (x_all[:, 5:] > person_conf).nonzero(as_tuple=False).T       
            x_all = torch.cat((box_all[i_all], x_all[i_all, j_all + 5, None], j_all[:, None].float()), 1)   
        else:  
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            x_all = x_all[(x_all[:, 5:6] == torch.tensor(classes, device=x_all.device)).any(1)] 


        if x.shape[0]:  
            x = x[x[:, 4].argsort(descending=True)]

            c = x[:, 5:6] * ( max_wh)  
            boxes, scores = x[:, :4] + c, x[:, 4]  
            i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  
                i = i[:max_det]
            output[xi] = x[i]
        else:
            output[xi] = []

        if x_all.shape[0]:

            output_all[xi] = x_all

        else:
            output_all[xi] = []

    return output, output_all

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

                        if cal_overlap(single_res[:4], group_single_res[:4], x1y1x2y2=True) >=0.05:
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


class FJN_YOLOR():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = None
        self.imgsz = 1280
        self.auto_size = 64
        self.init_model()

        self.conf_thres = 0.4   
        self.iou_thres = 0.5    
        self.classes = None     
        self.agnostic_nms = False    

        self.names = load_classes('ObjectDetector/yolor/data/coco.names')
        random.seed(1)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        pass

    def pre_proposse_img_cv(self, img_cv):
        img0 = copy.deepcopy(img_cv)  

        img = letterbox(img0, new_shape=self.imgsz, auto_size=self.auto_size)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img


    def detect_single(self, img_original):
        if isinstance(img_original, np.ndarray):
            img = self.pre_proposse_img_cv(img_original)
        else:
            img = img_original

        with torch.no_grad():
            pred = self.model(img, augment=False)[0]

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)[0]

            if pred is not None and len(pred):
                pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img_original.shape).round()

            if pred is not None:
                pred = pred.cpu().numpy()

            return pred

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
        det = self.detect_single(img_cv)
        if det is not None and len(det):

            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box(xyxy, img_cv, label=label, color=self.colors[int(cls)], line_thickness=3)
        return img_cv

    def draw_single_image_with_box(self, img_cv, box):
        img_cv = copy.deepcopy(img_cv)
        if box is not None and len(box):
            for *xyxy, conf, cls in box:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box(xyxy, img_cv, label=label, color=self.colors[int(cls)], line_thickness=3)
        return img_cv

    def get_labels(self, img_root, save_root):
        os.makedirs(save_root, exist_ok=True)

        img_name_ls = os.listdir(img_root)
        t = tqdm(total=len( img_name_ls ), ascii=True)
        for img_name in img_name_ls:
            if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png'):
                txt_name = img_name.replace('.'+img_name.split('.')[-1], '.txt')
                txt_path = os.path.join( save_root, txt_name )
                img_cv = cv2.imread( os.path.join( img_root, img_name ), 1 )

                gn = torch.tensor(img_cv.shape)[[1, 0, 1, 0]] 

                det = self.detect_single(img_cv)

                with open(txt_path, 'w+') as f:
                    for *xyxy, conf, cls in det:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh)) 
            t.update(1)
        t.close()
        pass

    def FindHiddenPerson(self, img, person_conf=0.05, overlap_thresh=0.4, remove_small_length=20, faster=False, shown=False):

 
        box_all_person, box_original_person = self.detect_person_and_original(img, person_conf=person_conf, faster=faster)

        if shown:
            print('box_all_person: ', box_all_person)
            print('box_original_person: ', box_original_person)
            cv2.imshow('box_original_person', self.draw_single_image_with_box(img, box_original_person))
            cv2.imshow('box_all_person', self.draw_single_image_with_box(img, box_all_person))


        box_all_person_overlap = []
        if len(box_original_person) > 0:
            for single_box_all_person in box_all_person:
                boo_is_overlap = False
                for single_box_original_person in box_original_person:

                    if cal_overlap(single_box_all_person[:4], single_box_original_person[:4], x1y1x2y2=True) >= overlap_thresh:
                        boo_is_overlap = True
                        break
                if not boo_is_overlap:
                    box_all_person_overlap.append(single_box_all_person)
        else:
            box_all_person_overlap = box_all_person

        if shown:
            print('box_all_person_overlap: ', box_all_person_overlap)
            cv2.imshow('box_all_person_overlap', self.draw_single_image_with_box(img, box_all_person_overlap))

        if len(box_all_person_overlap) > 0:
            new_all_person_xyxy_boxes = np.array(box_all_person_overlap)
            person_groups_xyxy = divid_person_group(new_all_person_xyxy_boxes)
            hidden_xyxy = np.zeros((len(person_groups_xyxy), 7))
            hidden_xyxy[:, :4] = person_groups_xyxy
        else:
            hidden_xyxy = np.array([])

        if shown:
            print('hidden_xyxy not remove small: ', hidden_xyxy)

        final_hidden_xyxy = []
        for item in hidden_xyxy:
            if (item[2]-item[0])>remove_small_length and (item[3]-item[1])>remove_small_length:
                final_hidden_xyxy.append(item)
        hidden_xyxy = np.array(final_hidden_xyxy)

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
            input_is_np = True if isinstance(img_original, np.ndarray) else False

            if input_is_np:
                img = self.pre_proposse_img_cv(img_original)
            else:
                img = img_original

            if not faster:  
                pred = self.model(img, augment=False)[0]
            else:
                pred = self.model(img, augment=False)[1]

            pred, pred_all = non_max_suppression_hidden(pred, person_conf, self.conf_thres, self.iou_thres, classes=0, faster=faster)
            pred, pred_all = pred[0], pred_all[0]

            if input_is_np:
                if pred is not None and len(pred):
                    pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img_original.shape).round()
                if pred_all is not None and len(pred_all):
                    pred_all[:, :4] = scale_coords(img.shape[2:], pred_all[:, :4], img_original.shape).round()
            else:
                if pred is not None and len(pred):
                    pred[:, :4] = pred[:, :4].round()
                if pred_all is not None and len(pred_all):
                    pred_all[:, :4] = pred_all[:, :4].round()

            if pred is not None and len(pred):
                pred = pred.cpu().detach().numpy()
            if pred_all is not None and len(pred_all):
                pred_all = pred_all.detach().cpu().numpy()

            return pred_all, pred

    def init_model(self):

        weights = 'ObjectDetector/yolor/checkpoints/yolor_p6_coco.pt'
        yolor_cfg = 'ObjectDetector/yolor/cfg/yolor_p6.cfg'
        model = Darknet(yolor_cfg).to(self.device)
        model.eval()

        try:
            ckpt = torch.load(weights, map_location=self.device)  
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            load_darknet_weights(model, weights)

        self.imgsz = check_img_size(self.imgsz, s=64) 
        self.model = model

        pass

    def set_hyp(self):
        hyp_path = 'ObjectDetector/yolor/data/hyp.scratch.1280.yaml'
        with open(hyp_path) as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)
            self.model.hyp = hyp
        pass

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))




