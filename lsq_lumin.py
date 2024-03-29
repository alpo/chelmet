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


def ical(x, I, m):
    '''x = s(t) (m-vector), f(t) (m-vector), k (n-vector)'''
    c = I[0, :]
    s = x[:m]
    f = x[m + 1: 2 * m + 1]
    k = x[2 * m:]
    kif = np.outer(f, k)
    cpkif = c + kif
    Ical = s * cpkif.T
    return Ical.T


def error_function(x, I, m):
    '''x = s(t) (m-vector), f(t) (m-vector), k (n-vector)'''
    Ical = ical(x, I, m)
    error = I - Ical
    ferror = error.flatten()
    return ferror


def error_function2(x, I, m):
    e = np.linalg.norm(error_function(x, I, m)) ** 2
    return e


def grad(x, I, m):
    Ical = ical(x, I, m)
    error = I - Ical
    grad = np.zeros_like(x)
    c = I[0, :]
    s = x[:m]
    f = x[m + 1: 2 * m + 1]
    k = x[2 * m:]

    ecik = error * (c + np.outer(f, k))
    dedsj = 2 * ecik.sum(axis=1)
    grad[:m] = -dedsj

    ek = error * k
    sek = ek.sum(axis=1)
    dedfj = 2 * s * sek
    grad[m + 1: 2 * m + 1] = -dedfj

    esf = error.T * s * f
    dedki = 2 * esf.sum(axis=1)
    grad[2 * m:] = -dedki
    return grad

for job in jobs:

    infn = job['infn']
    logger.info('Input file %s', infn)

    txtfn = 'roi_bri_' + infn[:-4] + '.txt'
    data = np.loadtxt(txtfn)
    logger.debug(data.shape)

    gamma = 1
    step = 2

    vid_time = data[:600:step, 0]
    #I0 = data[:600:step, 1] ** gamma
    area = data[:600:step, 2]
    I = (data[:600:step, 3:] ** gamma) / (255.0 ** (gamma - 1))
    m = vid_time.shape[0]
    n = I.shape[1]

    x0fn = 'x0_' + infn[:-4] + '.txt'
    x0 = None
    if os.path.isfile(x0fn):
        x0 = np.loadtxt(x0fn)
    if x0 is None or x0.shape[0] != 2 * m + n:
        x0 = np.ones(2 * m + n)
#        x0[m + 1: 2 * m + 1] = I0 / 10
    logger.debug(x0.shape)

    lb = np.zeros(2 * m + n)
    ub = np.ones(2 * m + n)
    ub[m:] = np.inf

    p = NLP(error_function2, x0, maxIter=1e5, maxFunEvals=1e7, lb=lb, ub=ub)
    p.args.f = (I, m)
    p.df = grad
    p.checkdf()
    r = p.solve('ralg', plot=1)
    x = r.xf
    logger.info(x)

    xfn = 'x_' + infn[:-4] + '.txt'

    np.savetxt(xfn, x)

    # p = NLP(error_function2, x, maxIter=1e4, maxFunEvals=1e6, lb=lb, ub=ub)
    # p.args.f = (I, m)
    # r = p.solve('ralg', plot=1)
    # x = r.xf
    # logger.info(x)

    # cons = []
    # for i in xrange(m):
    #     cons.append(lambda x, I, m_: x[i]) #s>=0
    #     cons.append(lambda x, I, m_: x[i + m]) #f>=0
    # for i in xrange(n):
    #     cons.append(lambda x, I, m_: x[i + 2 * m]) #k>=0

    # args = (I, m)

    # x = optimize.fmin_cobyla(error_function2, x0, cons, args)

    #x = optimize.leastsq(error_function, x0, args)[0]

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(vid_time, I)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(vid_time, I - ical(x, I, m))

    s = x[:m]
    f = x[m + 1: 2 * m + 1]
    k = x[2 * m:]

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(vid_time, s)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(vid_time, f)

    fig = plt.figure()
    plt.plot(vid_time, np.sqrt(area) / s)

    plt.show()
