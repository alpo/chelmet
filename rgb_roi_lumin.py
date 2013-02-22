# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv
import logging
import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen

from matplotlib import rc
rc('font', **{'family': 'serif'})
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble='\usepackage[utf8]{inputenc}')
rc('text.latex', preamble='\usepackage[russian]{babel}')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('roi_lumin')

jobs = [
    # {'infn': u'Коркино. Метеорит(360p_H.264-AAC).mp4',
    # 'fps': 29.97,
    # 'insert_list': [{'point': 19, 'correction': 77}],
    # 'show': False,
    # },
    # {'infn': u'МЕТЕОРИТ 15 02 2013г(360p_H.264-AAC).mp4',
    # 'fps': 25,
    # 'insert_list': [{'point': 18, 'correction': 65}],
    # 'show': False,
    # },
    # {'infn': u'ВЗРЫВ ЧЕЛЯБИНСК(360p_H.264-AAC)-cut.mp4',
    # 'fps': 20,
    # 'insert_list': [],
    # 'show': False,
    # },
    # {'infn': u'Армагеддон в Челябинске! (съемка камеры наблюдения)(360p_H.264-AAC).mp4',
    # 'fps': 25,
    # 'insert_list': [{'point': 70, 'correction': 87}],
    # 'show': True,
    # },
    # {'infn': u'Метеорит,Челябинск(360p_H.264-AAC)-cut.mp4',
    # 'show': True,
    # 'insert_list': [],
    # 'rois': [
    #     [110, 10, 30, 30],
    #     [110, 70, 30, 30],
    #     [110, 150, 30, 30],
    #     [210, 10, 30, 30],
    #     [320, 10, 30, 30],
    #     [390, 70, 30, 30],
    #     [460, 70, 30, 30]
    # ]
    # },

    {'infn': u'Метеорит в Каменске-Уральском 15.02.2013(360p_H.264-AAC).mp4',
    'insert_list': [],
    'show': True,
    'rois': [
        [10, 30, 30, 30],
        [10, 80, 30, 30],
        # [470, 0, 30, 30],
        # [510, 0, 30, 30],
        # [580, 140, 20, 20],
        [180, 110, 30, 30],
        [230, 110, 30, 30]
    ]
    },
    # {'infn': u'Падение метеорита(720p_H.264-AAC).mp4',
    # 'fps': 29.97,
    # 'insert_list': [],
    # 'show': False,
    # },
    # {'infn': u'метеорит над челябинском(360p_H.264-AAC).mp4',
    # 'fps': 22.745,
    # 'insert_list': [],
    # 'show': True,
    # },

    # {'infn': u'Падение метеорита в Челябинске Meteorite(360p_H.264-AAC).mp4',
    # 'insert_list': [],
    # 'show': True,
    # 'rois': [
    #    # [210, 150, 30, 30],
    #    # [270, 170, 30, 30],
    #    # [40, 60, 30, 30],
    #    # [330, 150, 30, 30],
    #    #  [180, 70, 30, 30],
    #     [10, 200, 20, 20],
    #     [320, 80, 30, 30],
    #     [60, 340, 20, 20],
    #     [550, 330, 20, 20]
    # ]
    # },
]

font1 = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, hscale=0.5, vscale=0.5,
    thickness=1)
font2 = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, hscale=0.5, vscale=0.5,
    thickness=2)

for job in jobs:

    infn = job['infn']
    logger.info('Input file %s', infn)

    capture = cv.CaptureFromFile(infn)
    fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
    logger.info('fps %.2f', fps)

    frame_num = 0
    job_rois = job['rois']
    brightness = []
    for i in enumerate(job_rois):
        brightness.append([])
        brightness.append([])
        brightness.append([])

    width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
    logger.debug(width)
    logger.debug(height)

#    grey = cv.CreateImage((width, height), cv.IPL_DEPTH_8U, 1)

    while True:
        frame = cv.QueryFrame(capture)
        if frame_num == 0:
            first_frame = cv.CloneImage(frame)
        if frame is None:
            break

        for i, roi in enumerate(job_rois):
            cv.SetImageROI(frame, tuple(roi))
            avg = cv.Avg(frame)
            brightness[3 * i].append(avg[0])
            brightness[3 * i + 1].append(avg[1])
            brightness[3 * i + 2].append(avg[2])
        frame_num += 1

    frame_nums = np.arange(frame_num)
#    fps = float(job['fps'])
    vid_time = frame_nums / fps
    logger.debug(vid_time.shape)
    logger.debug(len(brightness))
    logger.debug(len(brightness[0]))

    for insert in job['insert_list']:
        point = insert['point']
        correction = insert['correction']
        logger.info('Correction at %d - %ds', point, correction)
        vid_time[vid_time > point] += correction

    for i, roi in enumerate(job_rois):
        text = '%d' % i
        l, t, w, h = roi
        text_size = cv.GetTextSize(text, font2)[0]
        pos = (l + w / 2 - text_size[0] / 2,
            t + h / 2 + text_size[1] / 2)
        cv.PutText(first_frame, text, pos, font2, (0, 0, 0))
        cv.PutText(first_frame, text, pos, font1, (255, 255, 255))
        cv.PolyLine(first_frame,
            [[(l, t),
              (l + w, t),
              (l + w, t + h),
              (l, t + h),
              ]],
            is_closed=True, color=(0, 0, 0))

    jpgfn = 'rois_' + infn[:-4] + '.jpg'

    cv.SaveImage(jpgfn, first_frame)

    plt.figure()
    for i, _ in enumerate(brightness):
        point_num = i / 3
        component = ['r', 'g', 'b'][i % 3]
        label = '%d%s' % (point_num, component)
        plt.plot(vid_time, brightness[i], label=label)
    title = infn.replace('_', r'\_')
    plt.title(title)
    plt.legend()
    pngfn = 'rgb_roi_bri_' + infn[:-4] + '.png'
    plt.savefig(pngfn)

    if job['show']:
        plt.show()

    txtfn = 'rgb_roi_bri_' + infn[:-4] + '.txt'

    fmt = '%8.3f' + ' %6.1f' * (len(brightness) + 2)

    data = np.empty((len(vid_time), 1 + len(brightness)))
    data[:, 0] = vid_time
    for i, column in enumerate(brightness):
        data[:, i + 1] = column

    np.savetxt(txtfn, data, fmt='%8.3f')
