from utils.colors import skin_detector, get_colors, convert_rgb_to_names, find_hsv_group
from collections import Counter, defaultdict
from PIL import Image, ImageFile
from rembg.bg import remove
import numpy as np
import cv2, os, json, pickle, io

ImageFile.LOAD_TRUNCATED_IMAGES = True

max_conf = 0.85

H_range = {'H1': (0, 10), 'H2': (10, 20), 'H3': (20, 30), 'H4': (30, 40), 'H5': (40, 50), 'H6': (50, 60),
               'H7': (60, 70), 'H8': (70, 80), 'H9': (80, 90), 'H10': (90, 100), 'H11': (100, 110), 'H12': (110, 120),
               'H13': (120, 130), 'H14': (130, 140), 'H15': (140, 150), 'H16': (150, 160), 'H17': (160, 170),
               'H18': (170, 180)}
S_range = {'S1': (0, 15), 'S2': (15, 32), 'S3': (32, 45), 'S4': (45, 70), 'S5': (70, 91), 'S6': (91, 116),
               'S7': (116, 128), 'S8': (128, 162), 'S9': (162, 188), 'S10': (188, 206), 'S11': (206, 230),
               'S12': (230, 256)}
V_range = {'V1': (0, 25), 'V2': (25, 52), 'V3': (52, 75), 'V4': (75, 101), 'V5': (101, 131), 'V6': (131, 167),
               'V7': (167, 206), 'V8': (206, 256)}

female_clothes = ['full_cami_tops', 'full_tube_tops', 'regular_sleeveless_tops', 'half_tank_tops', 'half_cami_tops',
                      'floor_length_skirt', 'knee_length_skirt', 'half_dress', 'maxi_dress', 'tunic_tops',
                      'half_tube_tops', 'sleeved_crop_tops', 'mini_skirt', 'blouse', 'lehenga', 'saree']

skin_exposing_clothes = ['normal_shorts', 'jeans_short', 'blouse', 'mini_skirt', 'sleeved_crop_tops',
                             'full_tank_tops', 'half_tube_tops',
                             'jeans_short', 'half_dress', 'full_tube_tops', 'full_cami_tops', 'half_shirt',
                             'full_shirt', 'maxi_dress', 'shoes', 'slippers', 'heel',
                             'half_tshirt', 'half_cami_tops', 'half_tank_tops', 'regular_sleeveless_tops']

def get_hsv_tag(h,s,v):
    hsv_tag = []
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
    return hsv_tag

def find_invalid_det(confidence_score, name):
    if name == 'male' or name == 'female':
        return True
    if (name == 'saree') and (confidence_score <= 0.88):
        return True
    if confidence_score <= max_conf:
        return True
    return False

def get_det(det_nm, al, im_new):
    x1, y1, x2, y2 = int(al[0]), int(al[1]), int(al[2]), int(al[3])
    temp_image = im_new[y1 + int(int(y2 - y1) * 0.23):y2, x1:x2]
    if det_nm in skin_exposing_clothes:
        temp_image = skin_detector(temp_image)
    white_pixels = np.logical_and(255 == temp_image[:, :, 0],
                                  np.logical_and(255 == temp_image[:, :, 1],
                                                 255 == temp_image[:, :, 2]))
    if np.sum(white_pixels) > (87 / 100 * temp_image.shape[0] * temp_image.shape[1]):
        return -1000, 0, 0, 0, 0, 0, 0, 0
    rgb_col, hsv_col, rgb_hsv_per = get_colors(temp_image)  # returns BGR
    if rgb_col[0] == (255, 255, 255):
        return -1000, 0, 0, 0, 0, 0, 0, 0
    dominant_color = rgb_col[0]
    bgr_equi = np.array([dominant_color[2], dominant_color[1], dominant_color[0]],dtype='uint8').reshape(1, 1, 3)
    h, s, v = cv2.cvtColor(bgr_equi, cv2.COLOR_BGR2HSV).squeeze()
    hsv_tag = get_hsv_tag(h,s,v)
    return h, s, v, hsv_tag, dominant_color, rgb_col, hsv_col, rgb_hsv_per


def generate_dict(analytics_, gender_dicts):
    dup_gen_cat = []
    dup_hsv = defaultdict(list)
    dup_gen_ind = defaultdict(list)

    hsv_c = [(int(item['features']['hGrp'][1:]), int(item['features']['sGrp'][1:]), int(item['features']['vGrp'][1:]))
             for item in analytics_]
    gen_cat = [str(item['features']['gender'] + item['category']) for item in analytics_]

    for k, v in dict(Counter(gen_cat)).items():
        if v > 1:
            dup_gen_cat.append(k)

    for e in dup_gen_cat:
        for i, c in enumerate(gen_cat):
            if c == e:
                dup_gen_ind[e].append(i)

    for k, v in dup_gen_ind.items():
        for ind in v:
            dup_hsv[k].append(hsv_c[ind])

    invalid_indx = []
    try:
        for dup in dup_gen_cat:
            for i, val in enumerate(dup_hsv[dup]):
                h_r = (val[0] - 2, val[0] + 3)
                s_r = (val[1] - 2, val[1] + 3)
                v_r = (val[2] - 2, val[2] + 3)
                if i < len(val) - 1:
                    for r_v in dup_hsv[dup][i + 1:]:
                        if val[1] == 1 and r_v[1] == 1:  # For S
                            invalid_indx.append(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                            dup_gen_ind[dup].remove(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                            dup_hsv[dup].remove(r_v)
                        elif (val[2] in range(1, 3)) and (r_v[2] in range(1, 3)):  # For V
                            invalid_indx.append(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                            dup_gen_ind[dup].remove(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                            dup_hsv[dup].remove(r_v)
                        elif (r_v[0] in range(h_r[0], h_r[1])) and (r_v[1] in range(s_r[0], s_r[1])) and (
                                r_v[2] in range(v_r[0], v_r[1])):
                            invalid_indx.append(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                            dup_gen_ind[dup].remove(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                            dup_hsv[dup].remove(r_v)

    except Exception as e:
        print("Error")
        print(e)
        pass

    new_analytics_ = []
    for i, val in enumerate(analytics_):
        if i in invalid_indx:
            continue
        new_analytics_.append(val)

    vnew_analytics_ = []
    for val in new_analytics_:
        if val['features']['normalizedTime'] < 0.03:
            continue
        if val['features']['objectCount'] < 2:
            continue
        vnew_analytics_.append(val)

    all_feat = []
    all_val = []
    dup_feat = []
    all_feat = [ele['features']['gender'] + ele['category'] for ele in vnew_analytics_]
    for ele in vnew_analytics_:
        all_val.append(
            ele['features']['gender'] + ele['category'] + '+' + ele['features']['hGrp'] + ele['features']['sGrp'] \
            + ele['features']['vGrp'])

    for k, v in dict(Counter(all_feat)).items():
        if v > gender_dicts[k[0]]:
            dup_feat.append(k)

    uniq_ind = set()
    uniq_feat = []
    if len(dup_feat) > 0:
        try:
            for feat in dup_feat:
                for i, vl in enumerate(all_val):
                    if vl.split('+')[0][0] == 'U':
                        continue
                    elif (feat == vl.split('+')[0]) and (dict(Counter(uniq_feat)).get(vl.split('+')[0], 0) < gender_dicts[vl.split('+')[0][0]]):
                        uniq_ind.add(i)
                        uniq_feat.append(vl.split('+')[0])
                    elif (feat != vl.split('+')[0]) and (dict(Counter(uniq_feat)).get(vl.split('+')[0], 0) < gender_dicts[vl.split('+')[0][0]]):
                        uniq_ind.add(i)
                        uniq_feat.append(vl.split('+')[0])
        except:
            print("error")
            pass

    uniq_list = sorted(list(uniq_ind))
    if len(uniq_list) == 0:
        uniq_list = [i for i, v in enumerate(vnew_analytics_)]

    fnew_analytics_ = []
    for i, val in enumerate(vnew_analytics_):
        if i in uniq_list:
            fnew_analytics_.append(val)

    rm = []
    for ele in fnew_analytics_:
        if (ele['features']['h'] == 0) and (ele['features']['s'] == 0):
            rm.append(ele)
        elif (ele['features']['gender'] == 'U'):
            rm.append(ele)

    if len(rm) > 0:
        for ele in rm:
            fnew_analytics_.remove(ele)

    return fnew_analytics_

def apply_rembg(im0s):
    im0s = cv2.cvtColor(im0s, cv2.COLOR_RGB2BGR)
    pil_im = Image.fromarray(im0s)
    byt = io.BytesIO()
    pil_im.save(byt, 'PNG')
    f_value = byt.getvalue()
    result_im = remove(f_value)
    # imgs = Image.open(io.BytesIO(result_im)).convert("RGBA")
    imgs = Image.open(io.BytesIO(result_im))
    imgs.load()  # required for png.split()
    background = Image.new("RGB", imgs.size, (255, 255, 255))
    background.paste(imgs, mask=imgs.split()[3])
    im0s = np.asarray(background)
    im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
    return im0s