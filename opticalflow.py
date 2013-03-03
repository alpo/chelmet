# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv
import logging
import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen
from scipy.io import wavfile

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
    #  'insert_list': [{'point': 19, 'correction': 77}],
    #  'cut': [19, 2000],
    #  'show': True,
    #  },
    # {'infn': u'МЕТЕОРИТ 15 02 2013г(360p_H.264-AAC).mp4',
    #  'insert_list': [{'point': 18, 'correction': 65}],
    #  'cut': [19, 2000],
    #  'show': False,
    #  },
    # {'infn': u'ВЗРЫВ ЧЕЛЯБИНСК(360p_H.264-AAC)-cut.mp4',
    #  'insert_list': [],
    #  'cut': [140, 2000],
    #  'show': False,
    #  },
    # {'infn': u'Армагеддон в Челябинске! (съемка камеры наблюдения)(360p_H.264-AAC).mp4',
    #  'insert_list': [{'point': 70, 'correction': 87}],
    #  'cut': [80, 120],
    #  'show': True,
    #  },
    {'infn': u'метеорит над челябинском(360p_H.264-AAC).mp4',
     'insert_list': [],
     'cut': [110, 135],
     'show': False,
     },
    # {'infn': u'Метеорит в Каменске-Уральском 15.02.2013(360p_H.264-AAC).mp4',
    #  'fps': 30,
    #  'insert_list': [],
    #  'show': False,
    #  },
    # {'infn': u'Падение метеорита(720p_H.264-AAC).mp4',
    #  'fps': 29.97,
    #  'insert_list': [],
    #  'show': False,
    #  },
    # {'infn': u'В челябинске упал самолет или метеорит, смотреть с 0_40(360p_H.264-AAC).mp4',
    #  'insert_list': [],
    #  'show': False,
    #  'cut': [120, 2000],
    #  },

]

# cv.NamedWindow('bbb')

for job in jobs:
    infn = job['infn']
    logger.info('Input file %s', infn)

    capture = cv.CaptureFromFile(infn)
    fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
    width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
    logger.debug(width)
    logger.debug(height)

    frame_num = 0
    frame = cv.CreateMat(height, width, cv.CV_8UC1)
    prev_frame = cv.CreateMat(height, width, cv.CV_8UC1)
    velx = cv.CreateMat(height, width, cv.CV_32FC1)
    vely = cv.CreateMat(height, width, cv.CV_32FC1)
    avgxs = []
    avgys = []
    frame_nums = []

    first_frame = False

    while True:
        rgb_frame = cv.QueryFrame(capture)
        if rgb_frame is None:
            break
        if 'cut' in job:
            if frame_num / fps < job['cut'][0]:
                frame_num += 1
                continue
            if frame_num / fps > job['cut'][1]:
                break
        cv.CvtColor(rgb_frame, frame, cv.CV_RGB2GRAY)
        if first_frame:
            first_frame = False
            cv.Copy(frame, prev_frame)
            continue
        # cv.CalcOpticalFlowLK(prev_frame, frame, (15, 15), velx, vely)
        cv.CalcOpticalFlowHS(prev_frame, frame, 0, velx, vely,
                             0.01,
                            (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS,
                             10, 0.01))
        # cv.ShowImage('bbb', velx)
        # cv.WaitKey(1000.0 / fps)
        avgx = cv.Avg(velx)[0]
        avgy = cv.Avg(vely)[0]
        avgxs.append(avgx)
        avgys.append(avgy)
        frame_nums.append(frame_num)

        cv.Copy(frame, prev_frame)

        frame_num += 1
        # if frame_num >= 30:
        #      break

    vid_time = np.array(frame_nums) / fps

    for insert in job['insert_list']:
        point = insert['point']
        correction = insert['correction']
        logger.info('Correction at %d - %ds', point, correction)
        vid_time[vid_time > point] += correction

    logger.debug(vid_time.shape)
    logger.debug(len(avgxs))
    logger.debug(len(avgys))

    magn = np.sqrt(np.array(avgxs) ** 2 + np.array(avgys) ** 2)

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(vid_time, avgxs)
    plt.title('X velocity')
    ax2 = fig.add_subplot(312, sharex=ax1, sharey=ax1)
    ax2.plot(vid_time, avgys)
    plt.title('Y velocity')
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(vid_time, magn)
    plt.title('Magnitude')

    title = infn.replace('_', r'\_')
    plt.suptitle(title)
    pngfn = 'velocity_' + infn[:-4] + '.png'
    plt.savefig(pngfn)

#    if job['show']:
    plt.show()

    txtfn = 'velocity_' + infn[:-4] + '.txt'

    np.savetxt(txtfn, np.column_stack((vid_time,
                                       avgxs,
                                       avgys,
                                       magn)),
               fmt='%8.3f')
