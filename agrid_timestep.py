# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('agrid')

jobs = [
    {'infn': u'Коркино. Метеорит(360p_H.264-AAC).txt',
    'lat': 54.8907,
    'lon': 61.3997
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
    {'infn': u'метеорит над челябинском(360p_H.264-AAC).txt',
    'lat': 55.0081,
    'lon': 61.2131
    },

]

deglat = 111.11
deglon = 111.11 * math.cos(55 / 180.0 * math.pi)

logger.info('Degree of lon %f', deglon)

zero_ll = dict(lat=55.121386, lon=61.468978)

km_step = 0.5

lat_km_grid, lon_km_grid, h_grid = np.mgrid[0:70:km_step,
            -50:50:km_step, 0:40:km_step]

logger.info('Grid size %d MB', lat_km_grid.size / 2 ** 20)

timecorr_step = 0.1
timecorr_grid = np.arange(-2, 2, timecorr_step)

snd_sum = np.zeros((lat_km_grid.shape[0], lat_km_grid.shape[1],
    lat_km_grid.shape[2], timecorr_grid.shape[0]))

snd_prod = np.ones((lat_km_grid.shape[0], lat_km_grid.shape[1],
    lat_km_grid.shape[2], timecorr_grid.shape[0]))

sigma = 4
n = int(4 * sigma) | 1
x = np.arange(n) - (n - 1) / 2.0
gauss_smooth = np.exp(-x ** 2 / sigma ** 2)

for job in jobs:
    logger.info('File %s', job['infn'])
    data = np.loadtxt(job['infn'])
    time = data[:, 0]
    bri = data[:, 1]
    snd = data[:, 2]
    snd /= np.max(snd)
    snd[snd < 0.7] = 0
    snd = np.convolve(snd, gauss_smooth, 'same')

    idx0 = np.argmax(bri)
    time0 = time[idx0]
    logger.info('Time0 %f', time0)
    time = time - time0

    dist_lat_grid = -(job['lat'] - zero_ll['lat']) * deglat - lat_km_grid
    dist_lon_grid = (job['lon'] - zero_ll['lon']) * deglon - lon_km_grid
    dist_h_grid = h_grid

    dist_grid = np.sqrt(dist_lat_grid ** 2 + dist_lon_grid ** 2 + dist_h_grid ** 2)
    time_grid = dist_grid / 0.313

    for icorr, time_corr in enumerate(timecorr_grid):
        rev_snd = np.interp(time_grid, time + time_corr, snd, 0, 0)

        snd_sum[:, :, :, icorr] += rev_snd
        snd_prod[:, :, :, icorr] *= rev_snd

logger.debug(np.min(snd_sum))
logger.debug(np.max(snd_sum))
logger.debug(np.min(snd_prod))
logger.debug(np.max(snd_prod))
logger.debug(snd_prod.shape)

lonlat_extent = [-50 / deglon + 61.468978,
                50 / deglon + 61.468978,
                -70 / deglat + 55.121386,
                55.121386]

heightlat_extent = [0,
                40,
                70 / deglat + 61.468978,
                61.468978]

heightlon_extent = [0,
                40,
                -50 / deglon + 55.121386,
                50 / deglon + 55.121386]


time_integr = np.sum(np.sum(np.sum(snd_prod, axis=0), axis=0), axis=0)
# все индексы схлопнулись по очереди остался тот который был 4
logger.debug(timecorr_grid)
logger.debug(time_integr)

best_idx = np.argmax(time_integr)

plt.figure()
plt.plot(timecorr_grid, time_integr)

plt.figure()
plt.imshow(np.sum(snd_sum[:, :, :, best_idx], axis=2), extent=lonlat_extent)
plt.xlabel('E')
plt.ylabel('N')
plt.title('sum integrated by height')
plt.savefig('sum_integrated_by_height.png')
plt.figure()
plt.imshow(np.sum(snd_sum[:, :, :, best_idx], axis=1), extent=[0, 40, 70, 0])
plt.xlabel('height, km')
plt.ylabel('km to south from (55.121386,61.468978)')
plt.title('sum integrated by longitude')
plt.savefig('sum_integrated_by_longitude.png')
plt.figure()
plt.imshow(np.sum(snd_sum[:, :, :, best_idx], axis=0), extent=[0, 40, -50, 50])
plt.xlabel('height, km')
plt.ylabel('km to east from (55.121386,61.468978)')
plt.title('sum integrated by latitude')
plt.savefig('sum_integrated_by_latitude.png')

int_prod = np.sum(snd_prod, axis=3)

plt.figure()
plt.imshow(np.sum(int_prod, axis=2), 
    cmap=plt.gray(), extent=lonlat_extent)
plt.xlabel('E')
plt.ylabel('N')
plt.title('prod integrated by height')
plt.savefig('prod_integrated_by_height.png')
plt.figure()
plt.imshow(np.sum(int_prod, axis=1), 
    cmap=plt.gray(), extent=[0, 40, 70, 0])
plt.xlabel('height, km')
plt.ylabel('km to south from (55.121386,61.468978)')
plt.title('prod integrated by longitude')
plt.savefig('prod_integrated_by_longitude.png')
plt.figure()
plt.imshow(np.sum(int_prod, axis=0), 
    cmap=plt.gray(), extent=[0, 40, -50, 50])
plt.xlabel('height, km')
plt.ylabel('km to east from (55.121386,61.468978)')
plt.title('prod integrated by latitude')
plt.savefig('prod_integrated_by_latitude.png')

# plt.figure()
# plt.imshow(np.sum(snd_prod[:, :, :, best_idx], axis=2), 
#     cmap=plt.gray(), extent=lonlat_extent)
# plt.xlabel('E')
# plt.ylabel('N')
# plt.title('prod integrated by height')
# plt.savefig('prod_integrated_by_height.png')
# plt.figure()
# plt.imshow(np.sum(snd_prod[:, :, :, best_idx], axis=1), 
#     cmap=plt.gray(), extent=[0, 40, 70, 0])
# plt.xlabel('height, km')
# plt.ylabel('km to south from (55.121386,61.468978)')
# plt.title('prod integrated by longitude')
# plt.savefig('prod_integrated_by_longitude.png')
# plt.figure()
# plt.imshow(np.sum(snd_prod[:, :, :, best_idx], axis=0), 
#     cmap=plt.gray(), extent=[0, 40, -50, 50])
# plt.xlabel('height, km')
# plt.ylabel('km to east from (55.121386,61.468978)')
# plt.title('prod integrated by latitude')
# plt.savefig('prod_integrated_by_latitude.png')

#plt.imshow(rev_snd)
plt.show()

