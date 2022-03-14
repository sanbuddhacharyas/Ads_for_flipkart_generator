import cv2,os,time
import numpy as np
from collections import defaultdict, Counter


def find_gender(det, names):
    # Dictionaries for clothing and gender classification task
    m_count = 0
    f_count = 0
    cloth_center_dict = {}  # for storing center of the detected cloth
    male_range = {}  # for storing the x-axis range of male
    female_range = {}  # for storing the x-axis range of female
    cloth_gender = {}  # for storing the cloth and to which gender it belongs
    f_d = defaultdict(list)

    for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
        al = [x.item() for x in xyxy]  # gets list of bboxes
        x1, y1, x2, y2 = int(al[0]), int(al[1]), int(al[2]), int(al[3])

        cent_cloth = (int(x1) + (int(x2) - int(x1)) // 2, int(y1) + (int(y2) - int(y1)) // 2)
        tl, tr = (int(x1), int(y1)), (int(x2), int(y1))
        box_wt = int(x2) - int(x1)
        padding = int(0.1 * box_wt)
        tl, tr = (int(x1) + padding, int(y1)), (int(x2) - padding, int(y1))

        if names[int(cls)] == 'male':
            m_count += 1
            male_range[i] = (tl[0], tr[0])
        if names[int(cls)] == 'female':
            f_count += 1
            female_range[i] = (tl[0], tr[0])

        if not names[int(cls)] == 'male' and not names[int(cls)] == 'female':
            cloth_center_dict[i] = cent_cloth

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
            if len(f_d[ind]) > 1:
                cloth_gender[ind] = 'U'
            elif f_d[ind] == ['male']:
                cloth_gender[ind] = 'M'
            elif f_d[ind] == ['female']:
                cloth_gender[ind] = 'F'
            else:
                cloth_gender[ind] = 'U'
        except:
            cloth_gender[ind] = 'U'

    return m_count, f_count, cloth_gender

    
    

