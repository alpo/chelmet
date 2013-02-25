# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from subprocess import Popen
from openopt import NLP

from matplotlib import rc
rc('font', **{'family': 'serif'})
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('text.latex', preamble='\usepackage[utf8]{inputenc}')
rc('text.latex', preamble='\usepackage[russian]{babel}')

logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('lsq_lumin')

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
    # },

    {'infn': u'Метеорит в Каменске-Уральском 15.02.2013(360p_H.264-AAC).mp4',
    'show': True,
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
    # 'show': True,
    # },
]


def ical(x, I, m, penl):
    '''x = s(t) (m-vector), f(t) (m-vector), k (n-vector)'''
    c = I[0, :]
    s = x[:m]
    f = x[m + 1: 2 * m + 1]
    k = x[2 * m:]
#    sat = x[-1]
    kif = np.outer(f, k)
    cpkif = c + kif
    Ical = s * cpkif.T
#    Ical[Ical > sat] = sat
    return Ical.T


def error_function(x, I, m, penl):
    '''x = s(t) (m-vector), f(t) (m-vector), k (n-vector)'''
    Ical = ical(x, I, m, penl)
    error = I - Ical
    weight = np.dstack((5 * I,
        5 * (1 - I),
        np.ones_like(I)))
    weight = np.min(weight, axis=2)
#    error *= weight
    ferror = error.flatten()
    return ferror


def error_function2(x, I, m, penl):
    e = np.linalg.norm(error_function(x, I, m, penl)) ** 2
    s = x[:m]
    f = x[m + 1: 2 * m + 1]
    e1 = np.linalg.norm(np.diff(s, 2)) ** 2
    e2 = np.linalg.norm(np.diff(f, 2)) ** 2
    e = e + penl * e1 + penl * e2
    return e


def grad(x, I, m, penl):
    Ical = ical(x, I, m, penl)
    error = I - Ical
    weight = np.dstack((5 * I,
        5 * (1 - I),
        np.ones_like(I)))
    weight = np.min(weight, axis=2)
#    error *= weight
    grad = np.zeros_like(x)
    c = I[0, :]
    s = x[:m]
    f = x[m + 1: 2 * m + 1]
    k = x[2 * m:]

    ecik = error * (c + np.outer(f, k))
    dedsj = -2 * ecik.sum(axis=1)

    dedsj[0] += np.dot(s[0:3], [2, -4, 2]) * penl
    dedsj[1] += np.dot(s[0:4], [-4, 10, -8, 2]) * penl
    for i in xrange(2, len(dedsj) - 2):
        dedsj[i] += np.dot(s[i - 2:i + 3], [2, -8, 12, -8, 2]) * penl
    dedsj[-2] += np.dot(s[-4:], [2, -8, 10, -4]) * penl
    dedsj[-1] += np.dot(s[-3:], [2, -4, 2]) * penl

    grad[:m] = dedsj

    ek = error * k
    sek = ek.sum(axis=1)
    dedfj = -2 * s * sek

    dedfj[0] += np.dot(f[0:3], [2, -4, 2]) * penl
    dedfj[1] += np.dot(f[0:4], [-4, 10, -8, 2]) * penl
    for i in xrange(2, len(dedfj) - 2):
        dedfj[i] += np.dot(f[i - 2:i + 3], [2, -8, 12, -8, 2]) * penl
    dedfj[-2] += np.dot(f[-4:], [2, -8, 10, -4]) * penl
    dedfj[-1] += np.dot(f[-3:], [2, -4, 2]) * penl

    grad[m + 1: 2 * m + 1] = dedfj

    esf = error.T * s * f
    dedki = -2 * esf.sum(axis=1)
    grad[2 * m:] = dedki
    return grad

for job in jobs:

    infn = job['infn']
    logger.info('Input file %s', infn)

    txtfn = 'rgb_roi_bri_' + infn[:-4] + '.txt'
    data = np.loadtxt(txtfn)
    logger.debug(data.shape)

#    gamma = 2.2
    step = 2

    vid_time = data[:500:step, 0]
    Isrgb = data[:500:step, 1:] / 255.0
    a = 0.055
    I = ((Isrgb + a) / (1 + a)) ** 2.4
    I[Isrgb <= 0.04045] = Isrgb[Isrgb <= 0.04045] / 12.92
#    I = Isrgb ** 2.2

    m = vid_time.shape[0]
    n = I.shape[1]

    x0fn = 'rgb_x0_' + infn[:-4] + '.txt'
    x0 = None
    if os.path.isfile(x0fn):
        x0 = np.loadtxt(x0fn)
    if x0 is None or x0.shape[0] != 2 * m + n:
        #x0 = np.random.randn(2 * m + n)
        x0 = np.ones(2 * m + n)
    logger.debug(x0.shape)

    lb = np.zeros(2 * m + n)
    ub = np.ones(2 * m + n)
    ub[m:] = np.inf

    penl = 0.1

    p = NLP(error_function2, x0, ftol=1e-10, maxIter=1e5, maxFunEvals=1e7, lb=lb, ub=ub)
    p.args.f = (I, m, penl / step ** 2)
    p.df = grad
    p.checkdf()
    r = p.solve('ralg', plot=1)
    x = r.xf
    logger.info(x)

    xfn = 'rgb_x_' + infn[:-4] + '.txt'

    np.savetxt(xfn, x)

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.plot(vid_time, I)
    ax2 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    ax2.plot(vid_time, I - ical(x, I, m, penl))

    s = x[:m]
    f = x[m + 1: 2 * m + 1]
    k = x[2 * m:]

#    fig = plt.figure()

    ax1 = fig.add_subplot(222)
    ax1.plot(vid_time, s)
    ax2 = fig.add_subplot(224, sharex=ax1)
    ax2.plot(vid_time, f)

    plt.show()
