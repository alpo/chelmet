# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen
from scipy.io import wavfile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('agrid')

jobs = [
    {'infn': u'Коркино. Метеорит(360p_H.264-AAC).txt',
    'lat': 54.90,
    'lon': 61.41
    },
    {'infn': u'МЕТЕОРИТ 15 02 2013г(360p_H.264-AAC).txt',
    'lat': 54.76,
    'lon': 61.33
    },
    {'infn': u'ВЗРЫВ ЧЕЛЯБИНСК(360p_H.264-AAC)-cut.txt',
    'lat': 55.149944,
    'lon': 61.363964
    },
    {'infn': u'Армагеддон в Челябинске! (съемка камеры наблюдения)(360p_H.264-AAC).txt',
    'lat': 55.121386,
    'lon': 61.468978
    },
]

deglat = 111.11
deglon = 111.11 * math.cos(61.4 / 180.0 * math.pi)

logger.info('Degree of lon %f', deglon)

zero_ll = dict(lat=55.121386, lon=61.468978)

km_step = 0.5

lat_km_grid, lon_km_grid, h_grid = np.mgrid[0:70:km_step,
            -50:50:km_step, 0:40:km_step]

logger.info('Grid size %d MB', lat_km_grid.size / 2 ** 20)

snd_sum = np.zeros_like(lat_km_grid)
snd_prod = np.zeros_like(lat_km_grid)

for job in jobs:
    data = np.loadtxt(job['infn'])
    time = data[:, 0]
    bri = data[:, 1]
    snd = data[:, 2]
    snd /= np.max(snd)

    idx0 = np.argmax(bri)
    time0 = time[idx0]
    logger.info('Time0 %f', time0)
    time = time - time0

    dist_lat_grid = -(job['lat'] - zero_ll['lat']) * deglat - lat_km_grid
    dist_lon_grid = (job['lon'] - zero_ll['lon']) * deglon - lon_km_grid
    dist_h_grid = h_grid

    dist_grid = np.sqrt(dist_lat_grid ** 2 + dist_lon_grid ** 2 + dist_h_grid ** 2)
    time_grid = dist_grid / 0.32

    #for i in xrange(time_grid.shape[2]):

    rev_snd = np.interp(time_grid, time, snd, 0, 0)

    logger.debug(lat_km_grid.shape)
    logger.debug(lon_km_grid.shape)
    logger.debug(np.min(rev_snd))
    logger.debug(np.max(rev_snd))
    snd_sum += rev_snd
    snd_prod *= rev_snd

#plt.imshow(lat_km_grid[:,:,0], lon_km_grid[:,:,0], rev_snd[:,:,0])
#plt.imshow(rev_snd[:,:,0], extent=[-50, 50, 0, 50])
logger.debug(np.min(snd_sum))
logger.debug(np.max(snd_sum))
logger.debug(np.min(snd_prod))
logger.debug(np.max(snd_prod))

plt.figure()
plt.imshow(np.sum(snd_sum, axis=2), extent=[-50, 50, 70, 0])
plt.xlabel('km to east from (55.121386,61.468978)')
plt.ylabel('km to south from (55.121386,61.468978)')
plt.title('sum integrated by height')
plt.savefig('sum_integrated_by_height.png')
plt.figure()
plt.imshow(np.sum(snd_sum, axis=1), extent=[0, 40, 70, 0])
plt.xlabel('height, km')
plt.ylabel('km to south from (55.121386,61.468978)')
plt.title('sum integrated by longitude')
plt.savefig('sum_integrated_by_longitude.png')
plt.figure()
plt.imshow(np.sum(snd_sum, axis=0), extent=[0, 40, -50, 50])
plt.xlabel('height, km')
plt.ylabel('km to east from (55.121386,61.468978)')
plt.title('sum integrated by latitude')
plt.savefig('sum_integrated_by_latitude.png')
#plt.axis('image')

#plt.imshow(rev_snd)
plt.show()

