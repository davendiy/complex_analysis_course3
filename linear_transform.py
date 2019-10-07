#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from abc import abstractmethod, ABCMeta


def is_inf(z: complex):
    return abs(z) == float('inf')


def show_complex_point(z, fig, color):
    if is_inf(z):
        return

    if fig:
        fig.scatter([z.real], [z.imag], color=color)
    else:
        plt.scatter([z.real], [z.imag], color=color)


class Figure(metaclass=ABCMeta):

    @abstractmethod
    def show(self, bound_ext=10, step=1e-3, fig=None, **plt_kwargs):
        pass


class _Circle(Figure):

    def __init__(self, center: complex, radius: float):
        assert radius, 'Radius of circle must be non-zero.'
        self._c = center
        self._r = radius

    def show(self, step=1e-3, fig=None, **plt_kwargs):
        show_complex_point(self._c, fig, plt_kwargs.get('color', 'b'))
        angles = np.arange(0, 2 * np.pi, step)
        x = self._c.real + self._r * np.cos(angles)
        y = self._c.imag + self._r * np.sin(angles)
        if fig:
            fig.plot(x, y, **plt_kwargs)
        else:
            plt.plot(x, y, **plt_kwargs)


class _Line:

    def __init__(self, a: complex, b: complex):
        if a == b:
            raise ValueError("Points for line defining must be different.")

        assert not is_inf(a) and not is_inf(b), 'Infinity points for line.'

        self._a = a
        self._b = b

    def show(self, bound_ext=10, step=1e-3, fig=None, **plt_kwargs):
        show_complex_point(self._a, fig, plt_kwargs.get('color', 'b'))
        show_complex_point(self._b, fig, plt_kwargs.get('color', 'b'))

        x1, x2 = self._a.real, self._b.real
        y1, y2 = self._a.imag, self._b.imag

        if x2 - x1:
            x = np.arange(x1 - bound_ext, x2 + bound_ext, step)
            y = (x - x1) * (y2 - y1) / (x2 - x1) + y1
        else:
            y = np.arange(y1 - bound_ext, y2 + bound_ext, step)
            x = (y - y1) * (x2 - x1) / (y2 - y1) + x1

        if fig:
            fig.plot(x, y, **plt_kwargs)
        else:
            plt.plot(x, y, **plt_kwargs)


class ComplexCircle(_Line, _Circle):

    LINE = 1
    CIRCLE = 2

    def __init__(self, a: complex, b: complex, c: complex):
        if (is_inf(a) and is_inf(b) or
                is_inf(a) and is_inf(c) or
                is_inf(b) and is_inf(c)):
            raise ValueError("At least 2 points for circle definition "
                             "must be finite.")

        self._type = ComplexCircle.LINE
        if is_inf(a):
            _Line.__init__(self, b, c)
        elif is_inf(b):
            _Line.__init__(self, a, c)
        elif is_inf(c):
            _Line.__init__(self, a, b)
        elif ComplexCircle._is_points_on_line(a, b, c):
            _Line.__init__(self, a, b)
        else:
            tmp_center, tmp_radius = ComplexCircle._find_center_radius(a, b, c)
            self._type = ComplexCircle.CIRCLE
            self._points = [a, b, c]
            _Circle.__init__(self, tmp_center, tmp_radius)

    def show(self, bound_ext=10, step=1e-3, fig=None, **plt_kwargs):
        if self._type == ComplexCircle.CIRCLE:
            for point in self._points:
                show_complex_point(point, fig, plt_kwargs.get('color', 'b'))
            return _Circle.show(self, step, fig, **plt_kwargs)
        else:
            return _Line.show(self, bound_ext, step, fig, **plt_kwargs)

    @staticmethod
    def _is_points_on_line(a: complex, b: complex, c: complex, eps=1e-8):
        """ Checks if a, b, c are on the one line.

        Parameters
        ----------
        a, b, c - points on the complex plane.
        eps - accuracy

        Returns
        -------
        bool - True if square of (a, b, c) < eps
        """
        x1 = b - a
        x2 = c - a
        square = x1.real * x2.imag - x1.imag * x2.real
        return abs(square) < eps

    @staticmethod
    def _find_center_radius(a: complex, b: complex, c: complex):
        """ Calculating the center and radius of Circle, drown around
        the triangle (a, b, c).

        Center of such circle is intersection of perpendicular bisectors
        of triangle sides.

        Radius - distance between the center and any vertice of triangle.

        Parameters
        ----------
        a, b, c, - points on the complex plane

        Returns
        -------
        (center, radius), where
                            center - point on the complex plane
                            radius - float number
        """
        # v1(m1, n1) i v2(m2, n2) - direction vectors for (a, b) and (a, c)
        m1 = b.real - a.real
        n1 = b.imag - a.imag
        m2 = c.real - a.real
        n2 = c.imag - a.imag

        # (x1, y1) i (x2, y2) - mids of same sides of triangle
        x1 = (b.real + a.real) / 2
        y1 = (b.imag + a.imag) / 2
        x2 = (c.real + a.real) / 2
        y2 = (c.imag + a.imag) / 2

        # formula of x value of intersection of perpendicular bisectors
        x = (n1 * n2 * (y2 - y1) + x2 * m2 * n1 - x1 * m1 * n2) / \
            (n1 * m2 - n2 * m1)
        if n2 != 0:  # one of our vectors must have non-zero y-component
            y = (m2 * (x - x2)) / (-n2) + y2  # formula of y value of intersection
        else:
            y = (m1 * (x - x1)) / (-n1) + y1
        center = complex(x, y)

        radius = abs(a - center)
        return center, radius


class Region:
    pass


class Transformation(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, region: Region):
        pass


class FracLinearTransform(Transformation):

    def __init__(self, a: complex, b: complex, c: complex, d: complex):
        """ Fractional linear transformation of complex
        plane W = f(z) = (az + b) / (cz + d).

        Parameters
        ----------
        a, b, c, d - complex numbers - parameters of transformation.
        """
        pass


if __name__ == "__main__":
    test = _Circle(complex(2, 0), 5)
    test.show(label='test1, just circle')
    plt.legend()
    plt.show()

    test2 = _Line(complex(1, 0), complex(10, 0))
    test2.show(label='test2, just horizontal line')
    plt.legend()
    plt.show()

    test3 = _Line(complex(-2, 4), complex(-2, 10))
    test3.show(label='test3, just vertical line', color='b')

    test4 = ComplexCircle(complex(2, 3), complex(4, 5), complex(-5, 7))
    test4.show(label='test4, complex circle - just circle', color='r')

    test5 = ComplexCircle(complex(2, 3), complex(4, 5), float('inf'))
    test5.show(label='test5, complex circle with infinity', color='g')
    plt.legend()
    plt.show()
