# !/usr/bin/env python
# -*- coding: utf-8 -*-
import cv
import logging
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', **{'family': 'serif'})
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble='\usepackage[utf8]{inputenc}')
rc('text.latex', preamble='\usepackage[russian]{babel}')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('brightness')

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

    {'infn': u'Метеорит в Каменске-Уральском 15.02.2013(360p_H.264-AAC).mp4',
    'fps': 30,
    'insert_list': [],
    'show': True,
    'sky': {'left': 500,
            'top': 0,
            'width': 50,
            'height': 30},
    'trail': {'left': 0,
            'top': 0,
            'width': 460,
            'height': 220}
    },
    {'infn': u'Метеорит,Челябинск(360p_H.264-AAC).mp4',
    'fps': 30,
    'insert_list': [],
    'show': True,
    'sky': {'left': 0,
            'top': 0,
            'width': 50,
            'height': 50},
    },
    # {'infn': u'Падение метеорита(720p_H.264-AAC).mp4',
    # 'fps': 29.97,
    # 'insert_list': [],
    # 'show': False,
    # },
]

for job in jobs:
    infn = job['infn']
    logger.info('Input file %s', infn)

    capture = cv.CaptureFromFile(infn)

    frame_num = 0
    int_bri = []
    sky_bri = []
    m10_list = []
    m01_list = []
    while True:
        frame = cv.QueryFrame(capture)
        if frame is None:
            break
        # if frame_num < 450:
        #     frame_num += 1
        #     continue
        # if frame_num == 450:
        #     frame_num = 0
        grey = cv.CreateImage(cv.GetSize(frame),
            cv.IPL_DEPTH_8U, 1)
        cv.CvtColor(frame, grey, cv.CV_RGB2GRAY)

        avg = cv.Avg(grey)[0]
        int_bri.append(avg)

        sky = job['sky']
        cv.SetImageROI(grey, (sky['left'], sky['top'],
            sky['width'], sky['height']))
        sky_avg = cv.Avg(grey)[0]
        sky_bri.append(sky_avg)

        # trail = job['trail']
        # cv.SetImageROI(grey, (trail['left'], trail['top'],
        #     trail['width'], trail['height']))

        # moms = cv.Moments(grey)
        # m00 = cv.GetSpatialMoment(moms, 0, 0)
        # m10 = cv.GetSpatialMoment(moms, 1, 0)
        # m01 = cv.GetSpatialMoment(moms, 0, 1)
        # m10_list.append(m10/m00)
        # m01_list.append(m01/m00)
        cv.ResetImageROI(grey)

        # masked = cv.CreateImage(cv.GetSize(frame),
        #     cv.IPL_DEPTH_8U, 1)
        # cv.Threshold(grey, masked, 200, 0,
        #     cv.CV_THRESH_TOZERO_INV)
        # thr_avg = cv.Avg(masked)[0]
        # thr_int_bri.append(avg)


        # if frame_num == 32:
        #     col1 = grey[:, 20]
        # if frame_num == 180:
        #     col2 = grey[:, 180]
        # if frame_num == 300:
        #     col3 = grey[:, 357]
        frame_num += 1

    frame_nums = np.arange(frame_num)
    vid_time = frame_nums / float(job['fps'])

    for insert in job['insert_list']:
        point = insert['point']
        correction = insert['correction']
        logger.info('Correction at %d - %ds', point, correction)
        vid_time[vid_time > point] += correction

    int_bri = np.array(int_bri)
    sky_bri = np.array(sky_bri)
    rel_bri = int_bri / sky_bri

    time_step = 0.1
    max_time = np.max(vid_time)
    interp_time = np.arange(0, max_time, time_step)
    interp_intbri = np.interp(interp_time, vid_time, int_bri)
    interp_skybri = np.interp(interp_time, vid_time, sky_bri)
    interp_relbri = np.interp(interp_time, vid_time, rel_bri)

    # plt.figure()
    # plt.plot(col1)
    # plt.plot(col2)
    # plt.plot(col3)

    # plt.figure()
    # greym = cv.CreateMat(cv.GetSize(frame)[1], cv.GetSize(frame)[0],
    #     cv.CV_8UC1)
    # cv.Copy(grey, greym)
    # arr = np.asarray(greym)
    # logger.debug(arr)
    # plt.imshow(arr, cmap=plt.gray())
    # plt.plot(m10_list, m01_list)

    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.plot(vid_time, int_bri,
        vid_time, sky_bri)
    plt.legend(('Integral brightness', 'Sky brightness'))
    plt.xlabel('Time, s')
    plt.subplot(132)
    plt.plot(vid_time, rel_bri)
    plt.legend(('Integral brightness relative  to sky',))
    plt.xlabel('Time, s')
    plt.subplot(133)
    plt.semilogy(vid_time, rel_bri)
    plt.legend(('Integral brightness relative  to sky',))
#    plt.plot(interp_time, interp_bri)
    plt.xlabel('Time, s')
    title = infn.replace('_', r'\_')
    plt.suptitle(title)
    pngfn = 'rel_' + infn[:-4] + '.png'
    plt.savefig(pngfn)

    txtfn = 'rel_' + infn[:-4] + '.txt'

    np.savetxt(txtfn, np.column_stack((interp_time,
        interp_intbri, interp_skybri, interp_relbri)), 
    fmt='%4.1f %10.2f %10.2f %10.2f')

    if job['show']:
        plt.show()
