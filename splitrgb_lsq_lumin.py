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


def unpack_x(x, nframes, nrois):
    ''' x=[sr sg sb fr fg fb kr kg kb]
    sizes  nf nf nf nf nf nf nr nr nr
    '''
    ''' x=[s fr fg fb kr kg kb]
    sizes  n nf nf nf nr nr nr
    '''
    assert(x.shape[0] == 4 * nframes + 3 * nrois)
    s = x[:nframes]
    f = x[nframes:4 * nframes].reshape(nframes, 3)
    k = x[4 * nframes:].reshape(nrois, 3)
    return s, f, k


def pack_x(s, f, k):
    ''' x=[sr sg sb fr fg fb kr kg kb]
    sizes  nf nf nf nf nf nf nr nr nr
    '''
    x = np.hstack((s.flatten(), f.flatten(), k.flatten()))
    return x


def ical(x, I, nframes, nrois, penl):
    '''x = s(t) (nframes-vector), f(t) (nframes-vector), k (n-vector)'''
    c = I[0, :]
    logger.debug(show_shape('c'))
    s, f, k = unpack_x(x, nframes, nrois)
#    sat = x[-1]
    logger.debug(show_shape('s'))
    logger.debug(show_shape('f'))
    logger.debug(show_shape('k'))
    Ical = np.empty((nframes, nrois, 3))
    for i in xrange(3):
        kif = np.outer(f[:, i], k[:, i])
        logger.debug(show_shape('kif'))
        cpkif = c[:, i] + kif
        Ical[:, :, i] = (s * cpkif.T).T
#    Ical[Ical > sat] = sat
    return Ical


def error_function(x, I, nframes, nrois, penl):
    '''x = s(t) (nframes-vector), f(t) (nframes-vector), k (n-vector)'''
    Ical = ical(x, I, nframes, nrois, penl)
    error = I - Ical
    # weight = np.dstack((5 * I,
    #     5 * (1 - I),
    #     np.ones_like(I)))
    # weight = np.min(weight, axis=2)
#    error *= weight
    ferror = error.flatten()
    return ferror


def error_function2(x, I, nframes, nrois, penl):
    e = np.linalg.norm(error_function(x, I, nframes, nrois, penl)) ** 2
    s, f, k = unpack_x(x, nframes, nrois)
    sd2 = np.diff(s, 2, axis=0)
    fd2 = np.diff(f, 2, axis=0)
    es = np.linalg.norm(sd2) ** 2
    ef = np.linalg.norm(fd2) ** 2
    e = e + penl * es + penl * ef
    return e


def grad(x, I, nframes, nrois, penl):
    Ical = ical(x, I, nframes, nrois, penl)
    error = I - Ical
    # weight = np.dstack((5 * I,
    #     5 * (1 - I),
    #     np.ones_like(I)))
    # weight = np.min(weight, axis=2)
#    error *= weight
    grad = np.zeros_like(x)
    c = I[0, :]
    s, f, k = unpack_x(x, nframes, nrois)

    logger.debug(show_shape('error'))
    logger.debug(show_shape('c'))
    logger.debug(show_shape('f'))
    logger.debug(show_shape('k'))
    ecik = np.zeros((nframes, nrois))
    for i in xrange(3):
        kif = np.outer(f[:, i], k[:, i])
        logger.debug(show_shape('kif'))
        cpkif = c[:, i] + kif
        logger.debug(show_shape('cpkif'))
        ecik += error[:, :, i] * (c[:, i] +
            np.outer(f[:, i], k[:, i]))
    dedsj = -2 * ecik.sum(axis=1)

    dedsj[0] += np.dot(s[0:3], [2, -4, 2]) * penl
    dedsj[1] += np.dot(s[0:4], [-4, 10, -8, 2]) * penl
    for i in xrange(2, len(dedsj) - 2):
        dedsj[i] += np.dot(s[i - 2:i + 3], [2, -8, 12, -8, 2]) * penl
    dedsj[-2] += np.dot(s[-4:], [2, -8, 10, -4]) * penl
    dedsj[-1] += np.dot(s[-3:], [2, -4, 2]) * penl

    grad[:nframes] = dedsj

    ek = np.zeros((nframes, nrois))
    for i in xrange(3):
        ek += error[:, :, i] * k[:, i]
    sek = ek.sum(axis=1)
    dedfj = -2 * s * sek

    # dedfj[0] += np.dot(f[0:3], [2, -4, 2]) * penl
    # dedfj[1] += np.dot(f[0:4], [-4, 10, -8, 2]) * penl
    # for i in xrange(2, len(dedfj) - 2):
    #     dedfj[i] += np.dot(f[i - 2:i + 3], [2, -8, 12, -8, 2]) * penl
    # dedfj[-2] += np.dot(f[-4:], [2, -8, 10, -4]) * penl
    # dedfj[-1] += np.dot(f[-3:], [2, -4, 2]) * penl

    grad[nframes + 1: 2 * nframes + 1] = dedfj

    esf = error.T * s * f
    dedki = -2 * esf.sum(axis=1)
    grad[2 * nframes:] = dedki
    return grad


def show_shape(varname):
    import inspect
    frame = inspect.currentframe()
    try:
        var = frame.f_back.f_locals[varname]
    finally:
        del frame
    return '%s shape %s' % (varname, var.shape)

for job in jobs:

    infn = job['infn']
    logger.info('Input file %s', infn)

    txtfn = 'rgb_roi_bri_' + infn[:-4] + '.txt'
    data = np.loadtxt(txtfn)
    logger.debug(show_shape('data'))

#    gamma = 2.2
    step = 2

    nrois = (data.shape[1] - 1) / 3

    decimated_data = data[:500:step, :]

    vid_time = decimated_data[:, 0]
    nframes = vid_time.shape[0]
    logger.debug('nframes %d nrois %d', nframes, nrois)
    x_size = 4 * nframes + 3 * nrois
    '''x = s(t) (nframes-vector), f(t) (nframes-vector), k (n-vector)'''

    Isrgb = np.empty((nframes, nrois, 3))
    logger.debug(show_shape('Isrgb'))
    Isrgb[:, :, 0] = decimated_data[:, 1:nrois + 1] / 255.0
    Isrgb[:, :, 1] = decimated_data[:, nrois + 1:2 * nrois + 1] / 255.0
    Isrgb[:, :, 2] = decimated_data[:, 2 * nrois + 1:] / 255.0

    # a = 0.055
    # I = ((Isrgb + a) / (1 + a)) ** 2.4
    # I[Isrgb <= 0.04045] = Isrgb[Isrgb <= 0.04045] / 12.92
    I = Isrgb ** 2.2

    x0fn = 'rgb_x0_' + infn[:-4] + '.txt'
    x0 = None
    if os.path.isfile(x0fn):
        x0 = np.loadtxt(x0fn)
    if x0 is None or x0.shape[0] != x_size:
        x0 = np.random.randn(x_size)
    logger.debug(show_shape('x0'))

    lb = np.zeros(x_size)
    ub = np.ones(x_size)
    ub[nframes:] = np.inf

    penl = 0.01

    p = NLP(error_function2, x0, maxIter=1e5, maxFunEvals=1e7, lb=lb, ub=ub)
    p.args.f = (I, nframes, nrois, penl / step ** 2)
    p.df = grad
    p.checkdf()
    r = p.solve('ralg', plot=1)
    x = r.xf
    logger.info(x)

    xfn = 'rgb_x_' + infn[:-4] + '.txt'

    np.savetxt(xfn, x)

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    iresh = I.reshape(nframes, nrois * 3)
    ax1.plot(vid_time, iresh)
    ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
    ierr = I - ical(x, I, nframes, nrois, penl)
    ierrresh = ierr.reshape(nframes, nrois * 3)
    ax2.plot(vid_time, ierrresh)

    s, f, k = unpack_x(x, nframes, nrois)

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(vid_time, s)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(vid_time, f)

    plt.show()
