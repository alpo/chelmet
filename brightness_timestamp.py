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

    # {'infn': u'Метеорит в Каменске-Уральском 15.02.2013(360p_H.264-AAC).mp4',
    # 'fps': 30,
    # 'insert_list': [],
    # 'show': False,
    # },
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

    {'infn': u'Падение метеорита в Челябинске Meteorite(360p_H.264-AAC).mp4',
    'insert_list': [],
    'show': False,
    },
    {'infn': u'Метеорит над Челябинской областью(360p_H.264-AAC).mp4',
    'insert_list': [],
    'show': False,
    },
    {'infn': u'вспышка над Челябинском(360p_H.264-AAC).mp4',
    'insert_list': [],
    'show': False,
    },
    {'infn': u'В челябинске упал самолет или метеорит, смотреть с 0_40(360p_H.264-AAC).mp4',
    'insert_list': [],
    'show': False,
    },
]

font1 = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0.0,
    1, 8)
font2 = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0.0,
    2, 8)

cv.NamedWindow('mark')

for job in jobs:

    infn = job['infn']
    logger.info('Input file %s', infn)

    capture = cv.CaptureFromFile(infn)
    fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
    logger.info('fps %.2f', fps)

    frame_num = 0
    brightness = []
    while True:
        frame = cv.QueryFrame(capture)
        if frame is None:
            break
        avg = cv.Avg(frame)
        avg = avg[0] + avg[1] + avg[2]
        brightness.append(avg)
        frame_num += 1

    frame_nums = np.arange(frame_num)
#    fps = float(job['fps'])
    vid_time = frame_nums / fps

    for insert in job['insert_list']:
        point = insert['point']
        correction = insert['correction']
        logger.info('Correction at %d - %ds', point, correction)
        vid_time[vid_time > point] += correction

    frame_max = np.argmax(brightness)
    time_max = vid_time[frame_max]
    bri_max = brightness[frame_max]
    logger.info('frame %d, time %.2f, brightness %.2g', frame_max,
        time_max, bri_max)

    capture = cv.CaptureFromFile(infn)
    avifn = 'timed_' + infn[:-4] + '.avi'
    width = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT)
    #fourcc = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FOURCC)
    fourcc = cv.CV_FOURCC('I', '4', '2', '0')
    logger.info(fourcc)
    writer = cv.CreateVideoWriter(avifn,
        fourcc, fps,
        (width, height))

    frame_num = 0
    while True:
        frame = cv.QueryFrame(capture)
        if frame is None:
            break
        text = '%8.3f' % (vid_time[frame_num] - time_max)
        cv.PutText(frame, text, (10, 30), font2, (0, 0, 0))
        cv.PutText(frame, text, (10, 30), font1, (255, 255, 255))
        cv.ShowImage('mark', frame)
        cv.WriteFrame(writer, frame)
        #cv.WaitKey(1000 / fps)
        frame_num += 1

    mp4fn = 'timed_' + infn[:-4] + '.mp4'
    cmd = 'ffmpeg -i "%s" -b 2000k "%s"' % (avifn, mp4fn)
    logger.info(cmd)
    p = Popen(cmd, shell=True)
    sts = os.waitpid(p.pid, 0)[1]

    plt.figure()
    plt.plot(vid_time, brightness)
    plt.plot(time_max, bri_max, 'o')
    plt.text(time_max + 5, bri_max + 5, '(%.2f, %.0f)' % (time_max, bri_max))
    title = infn.replace('_', r'\_')
    plt.title(title)
    pngfn = 'br2_' + infn[:-4] + '.png'
    plt.savefig(pngfn)

    if job['show']:
        plt.show()

    txtfn = 'br2_' + infn[:-4] + '.txt'

    np.savetxt(txtfn, np.column_stack((vid_time,
        brightness)), fmt='%4.3f %10.1f')
