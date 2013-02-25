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
    {'infn': u'Коркино. Метеорит(360p_H.264-AAC).mp4',
    'fps': 29.97,
    'insert_list': [{'point': 19, 'correction': 77}],
    'show': False,
    },
    {'infn': u'МЕТЕОРИТ 15 02 2013г(360p_H.264-AAC).mp4',
    'fps': 25,
    'insert_list': [{'point': 18, 'correction': 65}],
    'show': False,
    },
    {'infn': u'ВЗРЫВ ЧЕЛЯБИНСК(360p_H.264-AAC)-cut.mp4',
    'fps': 20,
    'insert_list': [],
    'show': False,
    },
    {'infn': u'Армагеддон в Челябинске! (съемка камеры наблюдения)(360p_H.264-AAC).mp4',
    'fps': 25,
    'insert_list': [{'point': 70, 'correction': 87}],
    'show': True,
    },

    {'infn': u'Метеорит в Каменске-Уральском 15.02.2013(360p_H.264-AAC).mp4',
    'fps': 30,
    'insert_list': [],
    'show': False,
    },
    {'infn': u'Падение метеорита(720p_H.264-AAC).mp4',
    'fps': 29.97,
    'insert_list': [],
    'show': False,
    },
    {'infn': u'метеорит над челябинском(360p_H.264-AAC).mp4',
    'fps': 22.745,
    'insert_list': [],
    'show': False,
    },
]

for job in jobs:
    infn = job['infn']
    logger.info('Input file %s', infn)

    capture = cv.CaptureFromFile(infn)

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
    vid_time = frame_nums / float(job['fps'])

    for insert in job['insert_list']:
        point = insert['point']
        correction = insert['correction']
        logger.info('Correction at %d - %ds', point, correction)
        vid_time[vid_time > point] += correction

    wavfn = infn[:-4] + '.wav'
    if not os.path.isfile(wavfn):
        cmd = 'mplayer -ao pcm:file="%s" -novideo "%s"' % (wavfn, infn)
        logger.info(cmd)
        p = Popen(cmd, shell=True)
        sts = os.waitpid(p.pid, 0)[1]
    cmd = 'sox "%s" o.wav bandpass 20 20 bandpass 20 20 bandpass 20 20' % wavfn
    logger.info(cmd)
    p = Popen(cmd, shell=True)
    sts = os.waitpid(p.pid, 0)[1]

    samplerate, data = wavfile.read('o.wav')
    length = data.shape[0]
    logger.info('%d %d', samplerate, length)

    chunk_seconds = 0.2
    chunk_samples = int(samplerate * chunk_seconds)
    logger.debug('chunk_samples %s', chunk_samples)

    snd_time = []
    snd_level = []
    for start_sample in xrange(0, length, chunk_samples):
        chunk = data[start_sample:start_sample + chunk_samples] / 32768.0
        level = np.linalg.norm(chunk)
        snd_level.append(level)
        time = (start_sample + chunk_samples / 2.0) / samplerate
        snd_time.append(time)

    snd_time = np.array(snd_time)
    snd_level = np.array(snd_level)

    for insert in job['insert_list']:
        point = insert['point']
        correction = insert['correction']
        logger.info('Correction at %d - %ds', point, correction)
        snd_time[snd_time > point] += correction

    time_step = 0.2
    max_time = np.max(snd_time)
    interp_time = np.arange(0, max_time, time_step)
    interp_bri = np.interp(interp_time, vid_time, brightness)
    interp_snd = np.interp(interp_time, snd_time, snd_level)

    plt.figure()
    #plt.plot(vid_time, brightness, snd_time, snd_level * 20)
    plt.plot(interp_time, interp_bri, interp_time, interp_snd * 20)
    title = infn.replace('_', r'\_')
    plt.title(title)
    pngfn = infn[:-4] + '.png'
    plt.savefig(pngfn)

    if job['show']:
        plt.show()

    txtfn = infn[:-4] + '.txt'

    np.savetxt(txtfn, np.column_stack((interp_time,
        interp_bri, interp_snd)), fmt='%4.1f %10.1f %10.1f')
