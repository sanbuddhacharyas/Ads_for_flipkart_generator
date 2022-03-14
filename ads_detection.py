from __future__ import print_function
import os, shutil
import argparse
import time
from pathlib import Path
import cv2, io
import torch
import torch.backends.cudnn as cudnn
import requests
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time, json, math, pickle, os, shutil
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
from tqdm import tqdm
from rembg.bg import remove
from skimage.color import rgb2lab, deltaE_cie76
from sklearn.cluster import KMeans
from PIL import Image, ImageFile
import binascii
import struct,shutil
from PIL import Image
import numpy as np
import scipy,time,os
import scipy.misc
import scipy.cluster
from collections import defaultdict, Counter
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


ImageFile.LOAD_TRUNCATED_IMAGES = True


weights = './20220313_flipkart_v2.pt'

raw_img_path  = './raw'  #folder path


csv_path = './flipkart_ads_.csv'  #file path

total_im = sorted(os.listdir(raw_img_path))

weights_person = 'yolov5m.pt'

# Load model
device = ''

dnn = False
imgsz = [640, 640]
half=False
device = select_device(device)

model = DetectMultiBackend(weights, device=device, dnn=dnn)
model1 = DetectMultiBackend(weights_person, device=device, dnn=dnn)

stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
stride1, names1, pt1, jit1, onnx1 = model1.stride, model1.names, model1.pt, model1.jit, model1.onnx

imgsz = check_img_size(imgsz, s=stride)  # check image size
# Half
half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA

if pt:
    model.model.half() if half else model.model.float()
    model1.model.half() if half else model1.model.float()
    
# Run inference
if pt and device.type != 'cpu':
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    model1(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model1.model.parameters())))  # warmup
    
        
    
    
@torch.no_grad()
def run_person(weights_person='yolov5m.pt',  # model1.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        conf_thres=0.4,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all model1s
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inferenc
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    save_img = False
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    final_person = []
    final_id = []

        
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride1, auto=pt1 and not jit1)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride1, auto=pt1 and not jit1)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    dt, seen = [0.0, 0.0, 0.0], 0
    
    def spl(a):
        if a == "":
            return ''
        return a.split(os.sep)[-1].split('.')[0]

    for img_ind, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        
        total_person = 0
        ad_image_name = spl(path)
        final_id.append(ad_image_name)

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model1(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # n is total objects detected of single class
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names1[int(c)]}{'s' * (n > 1)}, "  # add to string

                

                # Write results
                for inde, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    try:
                        if names1[int(cls)] == 'person':
                            total_person += 1

                    except:
                        continue

            LOGGER.info(f'{s} Done. ({t3 - t2:.3f}s)')
            
            
        if (total_person > 1):
            final_person.append('ff')
        elif total_person <= 1:
            final_person.append('nice')
    
    return final_id[0], final_person[0]

                



def VideoAnalytics_person(video_path):
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights_person', nargs='+', type=str, default='yolov5m.pt',
                                help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=video_path, help='source')  # file/folder, 0 for webcam
#         parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
#                             help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all model1s')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        opt = parser.parse_args(args=[])
#         opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#         check_requirements(exclude=('tensorboard', 'thop'))

        fid, fp = run_person(**vars(opt))
        return fid, fp

    except Exception as e:
        print(e)
        return None
    
    
    

@torch.no_grad()
def run(source='data/images',  # file/dir/URL/glob, 0 for webcam
        conf_thres=0.4,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    save_img = False
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    img_count = 0
    all_detected_names = []
    preds = []
    detected_images = []
    cloth_details = []
    unique_cloth_names = set()
    write_count = 0
    detected_cats = []
    detected_confs = []
    detected_bbox = []
    detected_genders = []
    detected_colors = []
    detected_percent = []
    detected_hsv = []
    detected_hsv_group = []
    # total_clothes_details_list = []
    # analytics = {'videoIndex':'', 'videoSource':'', 'analyticsInfo':[]}
    H_range = {'H1': (0, 10), 'H2': (10, 20), 'H3': (20, 30), 'H4': (30, 40), 'H5': (40, 50), 'H6': (50, 60),
               'H7': (60, 70),'H8': (70, 80), 'H9': (80, 90), 'H10': (90, 100), 'H11': (100, 110), 'H12': (110, 120),
               'H13': (120, 130),'H14': (130, 140), 'H15': (140, 150), 'H16': (150, 160), 'H17': (160, 170), 'H18': (170, 180)}
    S_range = {'S1': (0, 15), 'S2': (15, 32), 'S3': (32, 45), 'S4': (45, 70), 'S5': (70, 91), 'S6': (91, 116),
               'S7': (116, 128), 'S8': (128, 162), 'S9': (162, 188), 'S10': (188, 206), 'S11': (206, 230),
               'S12': (230, 256)}
    V_range = {'V1': (0, 25), 'V2': (25, 52), 'V3': (52, 75), 'V4': (75, 101), 'V5': (101, 131), 'V6': (131, 167),
               'V7': (167, 206), 'V8': (206, 256)}
    female_clothes = ['full_cami_tops', 'full_tube_tops', 'regular_sleeveless_tops', 'half_tank_tops', 'half_cami_tops',
                      'floor_length_skirt','knee_length_skirt', 'half_dress', 'maxi_dress', 'tunic_tops', 'half_tube_tops',
                      'sleeved_crop_tops','mini_skirt', 'blouse', 'lehenga', 'saree']

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    dt, seen = [0.0, 0.0, 0.0], 0
    
    def spl(a):
        if a == "":
            return ''
        return a.split(os.sep)[-1].split('.')[0]

    for img_ind, (path, im, im0s, vid_cap,s) in enumerate(dataset):
        
        ad_image_name = spl(path)
        cat_l = []
        conf_l = []
        gender_l = []
        bbox_l = []
        colors_l = []
        percent_l = []
        hsv_l = []
        hsv_group_l = []
        g_count = 0
        img_hei = im0s.shape[0]
        img_wid = im0s.shape[1]
        
        
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
            
        # Dictionaries for clothing and gender classification task
        cloth_center_dict = {}  # for storing center of the detected cloth
        male_range = {}  # for storing the x-axis range of male
        female_range = {}  # for storing the x-axis range of female
        cloth_gender = {}  # for storing the cloth and to which gender it belongs
        f_d = defaultdict(list)
        
        ad_image_human_seg = apply_rembg(im0s, model_name = 'u2net_human_seg')
#         ad_image_human_seg = cv2.cvtColor(ad_image_human_seg, cv2.COLOR_BGR2RGB)
        

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # img.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            im_new = im0.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # n is total objects detected of single class
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                for inn,(*xyxy, conf, cls) in enumerate(reversed(det)):
                    al = [x.item() for x in xyxy]  # gets list of bboxes
                    x1, y1, x2, y2 = int(al[0]), int(al[1]), int(al[2]), int(al[3])
                    cent_cloth = (int(x1) + (int(x2) - int(x1)) // 2, int(y1) + (int(y2) - int(y1)) // 2)
                    tl, tr = (int(x1), int(y1)), (int(x2), int(y1))
                    box_wt = int(x2) - int(x1)
                    padding = int(0.1 * box_wt)
                    tl, tr = (int(x1) + padding, int(y1)), (int(x2) - padding, int(y1))

                    if names[int(cls)] == 'male' and conf.item() > 0.4:
                        g_count += 1
                        male_range[inn] = (tl[0], tr[0])
                    if names[int(cls)] == 'female' and conf.item() > 0.4:
                        g_count += 1
                        female_range[inn] = (tl[0], tr[0])

                    if not names[int(cls)] == 'male' and not names[int(cls)] == 'female':
                        cloth_center_dict[inn] = cent_cloth
                
            
                for ind, cent in cloth_center_dict.items():
                    for m_gen, xm_range in male_range.items():
                        if cent[0] in range(xm_range[0], xm_range[1]):
                            # print(f'{item} is of {m_gen}')
                            f_d[ind].append('male')
                            break

                    for f_gen, xf_range in female_range.items():
                        if cent[0] in range(xf_range[0], xf_range[1]):
                            # print(f'{item} is of {f_gen}')
                            f_d[ind].append('female')
                            break

                
                for ind, c in cloth_center_dict.items():
                    try:
                        if len(f_d[ind])>1:
                            cloth_gender[ind] = 'U'
                        elif f_d[ind] == ['male']:
                            cloth_gender[ind] = 'M'
                        elif f_d[ind] == ['female']:
                            cloth_gender[ind] = 'F'
                        else:
                            cloth_gender[ind] = 'U'
                    except:
                        cloth_gender[ind] = 'U'
                
                if g_count>1:
                    break
                
                # Write results
                for inde, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    try:
                        if names[int(cls)] == 'male' or names[int(cls)] == 'female':
                            continue
                            
                        confidence_score = conf.item()
                        al = [x.item() for x in xyxy ] #gets list of bboxes
                        x1, y1, x2, y2 = int(al[0]), int(al[1]), int(al[2]), int(al[3])
                        nx1 = x1/img_wid
                        nx2 = x2/img_wid
                        ny1 = y1/img_hei
                        ny2 = y2/img_hei
                        
                        if (names[int(cls)] == 'shoes') or (names[int(cls)] == 'slippers') or (names[int(cls)] == 'heel'):
                            det_img = ad_image_human_seg[y1:y2, x1:x2]
                            white_pixels = np.logical_and(255 == det_img[:, :, 0],
                                  np.logical_and(255 == det_img[:, :, 1], 255 == det_img[:, :, 2]))
                            num_white = np.sum(white_pixels)
                            all_pixels = det_img.shape[0] * det_img.shape[1]
                            
                            if (num_white*100/all_pixels) >85:
                                ad_image_u2net =  apply_rembg(im_new, model_name = 'u2net')
#                                 ad_image_u2net = cv2.cvtColor(ad_image_u2net, cv2.COLOR_BGR2RGB)
                                det_img = ad_image_u2net[y1:y2, x1:x2]
                                
                            colors, percentt = get_colors(det_img)
                            hsv_lis = []
                            hsv_grp_lis = []
                            for dominant_color in colors:
                                hsv_tag = []
                                bgr_equi = np.array([dominant_color[2], dominant_color[1], dominant_color[0]],
                                                                                dtype='uint8').reshape(1, 1, 3)
                                h, s, v = cv2.cvtColor(bgr_equi, cv2.COLOR_BGR2HSV).squeeze()

                                for h_name, h_range in H_range.items():
                                    if h in range(h_range[0], h_range[1]):
                                        hsv_tag.append(h_name)
                                        break
                                for s_name, s_range in S_range.items():
                                    if s in range(s_range[0], s_range[1]):
                                        hsv_tag.append(s_name)
                                        break
                                for v_name, v_range in V_range.items():
                                    if v in range(v_range[0], v_range[1]):
                                        hsv_tag.append(v_name)
                                        break
                                hsv_lis.append(tuple([h,s,v]))
                                hsv_grp_lis.append(tuple(hsv_tag))
                            colors_l.append(colors)
                            percent_l.append(percentt)
                            hsv_l.append(hsv_lis)
                            hsv_group_l.append(hsv_grp_lis)
                            cat_l.append(str(names[int(cls)]))
                            conf_l.append(round(confidence_score,2))
                            gender_l.append(str(cloth_gender[inde]))
                            bbox_l.append((nx1,ny1,nx2,ny2))

                        else:
                            if confidence_score > 0.4:
                                if names[int(cls)] in female_clothes:
                                    cloth_gender[inde] = 'F'
                                det_img = ad_image_human_seg[y1:y2, x1:x2]
                                white_pixels = np.logical_and(255 == det_img[:, :, 0],
                                      np.logical_and(255 == det_img[:, :, 1], 255 == det_img[:, :, 2]))
                                num_white = np.sum(white_pixels)
                                all_pixels = det_img.shape[0] * det_img.shape[1]

                                if (num_white*100/all_pixels) >85:
                                    ad_image_u2net =  apply_rembg(im_new, model_name = 'u2net')
                                    det_img = ad_image_u2net[y1:y2, x1:x2]

                                colors, percentt = get_colors(det_img)
                                hsv_lis = []
                                hsv_grp_lis = []
                                for dominant_color in colors:
                                    hsv_tag = []
                                    bgr_equi = np.array([dominant_color[2], dominant_color[1], dominant_color[0]],
                                                                                    dtype='uint8').reshape(1, 1, 3)
                                    h, s, v = cv2.cvtColor(bgr_equi, cv2.COLOR_BGR2HSV).squeeze()

                                    for h_name, h_range in H_range.items():
                                        if h in range(h_range[0], h_range[1]):
                                            hsv_tag.append(h_name)
                                            break
                                    for s_name, s_range in S_range.items():
                                        if s in range(s_range[0], s_range[1]):
                                            hsv_tag.append(s_name)
                                            break
                                    for v_name, v_range in V_range.items():
                                        if v in range(v_range[0], v_range[1]):
                                            hsv_tag.append(v_name)
                                            break
                                    hsv_lis.append(tuple([h,s,v]))
                                    hsv_grp_lis.append(tuple(hsv_tag))
                                colors_l.append(colors)
                                percent_l.append(percentt)
                                hsv_l.append(hsv_lis)
                                hsv_group_l.append(hsv_grp_lis)
                                cat_l.append(str(names[int(cls)]))
                                conf_l.append(round(confidence_score,2))
                                gender_l.append(str(cloth_gender[inde]))
                                bbox_l.append((nx1,ny1,nx2,ny2))

                                
                                    
                    except Exception as e:
                        print(e)
                        continue
              
        
        detected_cats.append(cat_l)
        detected_colors.append(colors_l)
        detected_percent.append(percent_l)
        detected_hsv.append(hsv_l)
        detected_hsv_group.append(hsv_group_l)
        detected_confs.append(conf_l)
        detected_genders.append(gender_l)
        detected_bbox.append(bbox_l)
        
    return detected_cats, detected_confs, detected_genders, detected_bbox, detected_colors, detected_percent, detected_hsv, detected_hsv_group




#For list of images

def VideoAnalytics(video_path):
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', type=str, default=video_path, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        opt = parser.parse_args(args=[])
        #check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))
#         check_requirements(exclude=('tensorboard', 'thop'))

        
        cats, detected_c, detected_g, det_b, det_col, det_per, det_hsv,det_hsv_g = run(**vars(opt))
        return cats, detected_c, detected_g, det_b, det_col, det_per, det_hsv,det_hsv_g

    except Exception as e:
        print(e)
        return None
    
    
    
def convert_rgb_to_names(rgb_tuple):
    try:
        # a dictionary of all the hex and their respective names in css3
        css3_db = CSS3_HEX_TO_NAMES
        names = []
        rgb_values = []
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))

        kdt_db = KDTree(rgb_values)
        distance, index = kdt_db.query(rgb_tuple)

        return names[index]
    except:
        return 'None'

def get_colors(image, NUM_CLUSTERS=4, show_chart=False):
    try:
        im = Image.fromarray(image)
        im = im.resize((100, 100))      # optional, to reduce time
        ar = np.asarray(im)
        #ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        modified_image = ar.reshape(ar.shape[0]*ar.shape[1], 3).astype(float)
        modified_image = np.where(modified_image <= 252, modified_image, 255)

        ar = modified_image[(modified_image!=np.array([[255, 255, 255]])).all(axis=1)]

        if list(ar)==[]:
            return [(255, 255, 255)], [0]

        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

        vecs, dist = scipy.cluster.vq.vq(ar, codes)         

        counts, bins = np.histogram(vecs, len(codes))   
        dic = {tuple(v):k for k,v in zip(counts,codes)}
        dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}  #sorting based on max occurance
        col = [k for k,v in dic.items()]
        total_count= sum([v for k,v in dic.items()])
        per = [v*100/total_count for k,v in dic.items()]
        return col, per
    except Exception as e:
        print(e)
        return [(255, 255, 255)], [0]
    
    
    
def apply_rembg(im0s, model_name):
    pil_im = Image.fromarray(im0s)
    byt = io.BytesIO()
    pil_im.save(byt, 'PNG')
    f_value = byt.getvalue()
    result_im = remove(f_value, model_name = model_name)
    # imgs = Image.open(io.BytesIO(result_im)).convert("RGBA")
    imgs = Image.open(io.BytesIO(result_im))
    imgs.load()  # required for png.split()
    background = Image.new("RGB", imgs.size, (255, 255, 255))
    background.paste(imgs, mask=imgs.split()[3])
    imms = np.asarray(background)
    imms = cv2.cvtColor(imms, cv2.COLOR_BGR2RGB)
    return imms

    
#for list of images

# from tqdm.notebook import tqdm
def check_cat(cat,bbox):
    if cat == []:
        return 'nice'
    if len(cat) == 1:
        return 'nice'
    
    if (len(set(cat)) == 1) and ('slippers' in cat):
        return 'nice'
    elif (len(set(cat)) == 1) and ('shoes' in cat):
        return 'nice'
    elif (len(set(cat)) == 1) and ('heel' in cat):
        return 'nice'
    
    ind = [i for i,v in enumerate(cat) if v not in ['slippers','shoes','heel']]
    if len(ind) == 0:
        return 'nice'
    x1,y1,x2,y2 = bbox[ind[0]]
    img_width = int(x2-x1)
    center = x1 + int(img_width//2)
    cent_x1 = center - int(3/10*img_width)
    cent_x2 = center + int(3/10*img_width)
    main_x1_x2 = [cent_x1, cent_x2, center]
        
    for ct,bb in zip(cat,bbox):
        if ct in ['slippers','shoes','heel']:
            continue
        x1,y1,x2,y2 = bb
        if (main_x1_x2[0] in range(int(x1),int(x2))) or (main_x1_x2[1] in range(int(x1),int(x2))) or (main_x1_x2[2] in range(int(x1),int(x2))):
            continue
        else:
            return 'ff'
    return 'nice'  




cs = []
ih = []
dfs = []
dr = []
dx = []
dl = []
dpr = []
dv = []
dg = []
ds =[]
ffid= []
ffpp = []

total_im = sorted(total_im)
try:
    load_from_csv  = pd.read_csv(csv_path, usecols=['DetectedCategory', 'DetectedRGB', 'DetectedPercent', 'DetectedHSV', 'DetectedHSVGroup', 'DetectedConf', 'DetectedGender','Detectedbbox', 'Duplicates', 'ads_id', 'person'])
    processed_list = load_from_csv['ads_id'].to_list()
    print(processed_list)

except:
    processed_list = []
    pass

for ind, v in tqdm(enumerate(total_im)):
    if v.split('/')[-1].split('.')[0] in processed_list:
        continue
    raw_path = os.path.join(raw_img_path,v)
    print(raw_path)
    try:
        cates, detec_conf, detec_gender, detec_bbox, detec_col, detec_per, detec_hsv,detec_hsv_g = VideoAnalytics(raw_path)
        f_idd, f_pp = VideoAnalytics_person(raw_path)
    except:
        continue
    
    ffid.append(f_idd)
    ffpp.append(f_pp)
    ds.append(check_cat(cates[0], detec_bbox[0]))
    cs.append(cates[0])
    dfs.append(detec_conf[0])
    dr.append(detec_gender[0])
    dx.append(detec_bbox[0])
    dl.append(detec_col[0])
    dpr.append(detec_per[0])
    dv.append(detec_hsv[0])
    dg.append(detec_hsv_g[0])
    
    if (ind + 1)%1000==0:
        d = pd.DataFrame()
        d['DetectedCategory'] = cs
        d['DetectedRGB'] = dl
        d['DetectedPercent'] = dpr
        d['DetectedHSV'] = dv
        d['DetectedHSVGroup'] = dg 
        d['DetectedConf'] = dfs
        d['DetectedGender'] = dr
        d['Detectedbbox'] = dx
        d['Duplicates'] = ds
        d['ads_id'] = ffid
        d['person'] = ffpp
        d.to_csv(os.path.join(csv_path), index=False)
        


d = pd.DataFrame()
d['DetectedCategory'] = cs
d['DetectedRGB'] = dl
d['DetectedPercent'] = dpr
d['DetectedHSV'] = dv
d['DetectedHSVGroup'] = dg 
d['DetectedConf'] = dfs
d['DetectedGender'] = dr
d['Detectedbbox'] = dx
d['Duplicates'] = ds
d['ads_id'] = ffid
d['person'] = ffpp
d.to_csv(os.path.join(csv_path), index=False)








 

    
    
    
    

