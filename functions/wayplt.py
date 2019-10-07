#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 14.09.2019
# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

import numpy as np
import matplotlib.pyplot as plt


def test_curve(t):
    return complex((1 - t**2)/(1 + t**2), 2*t/(1 + t**2))


def find_bounds(curve_func, start=0, end=1, step=10e-5):
    """Find bounds of the complex curve in parametric form.

    Parameters
    ----------
    curve_func : callable
        Function of real single parameter t from the segment [a, b]
        that returns complex number.
    start : float
        Left bound of segment [a, b]. Default: 0
    end : float
        Right bound of segment [a, b]. Default: 1
    step : float
        Distanse between two neighbouring points in splitting of the
        segment [a, b] (some kind of accuracy).

    Returns
    -------
    tuple
        (<min real value>, <max real value>, <min imag value>, <max imag value>)
    """
    tmp = curve_func(start)
    max_real, max_imag = tmp.real, tmp.imag
    min_real, min_imag = max_real, max_imag
    for t in np.arange(start, end, step):
        tmp = curve_func(t)
        max_real = max(max_real, tmp.real)
        max_imag = max(max_imag, tmp.imag)
        min_real = min(min_real, tmp.real)
        min_imag = min(min_imag, tmp.imag)
    return min_real, max_real, min_imag, max_imag


def plot_complex_curve(curve_func, start=0, end=0, step=10e-5):
    """Plots a complex curve in parametric form at the complex plane.

    Parameters
    ----------
    curve_func : callable
        Function of real single parameter t from the segment [a, b]
        that returns complex number.
    start : float
        Left bound of segment [a, b]. Default: 0
    end : float
        Right bound of segment [a, b]. Default: 1
    step : float
        Distanse between two neighbouring points in splitting of the
        segment [a, b] (some kind of accuracy).

    Returns
    -------
    None
    """
    t = np.arange(start, end, step)
    z = np.vectorize(curve_func)(t)
    plt.plot(z.real, z.imag)
