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
    def show(self, bound_ext=10., step=1e-3, fig=None, **plt_kwargs):
        pass


class _Circle(Figure):

    def __init__(self, center: complex, radius: float):
        assert radius, 'Radius of circle must be non-zero.'
        self._c = center
        self._r = radius

    def show(self, *, show_points=True, step=1e-3, fig=None, **plt_kwargs):
        if show_points:
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

    def show(self, *, show_points=True, bound_ext=10., step=1e-3,
             fig=None, **plt_kwargs):
        if show_points:
            show_complex_point(self._a, fig, plt_kwargs.get('color', 'b'))
            show_complex_point(self._b, fig, plt_kwargs.get('color', 'b'))

        x1, x2 = self._a.real, self._b.real
        y1, y2 = self._a.imag, self._b.imag

        if x2 - x1:
            x = np.arange(x1 - bound_ext/2, x2 + bound_ext/2, step)
            y = (x - x1) * (y2 - y1) / (x2 - x1) + y1
        else:
            y = np.arange(y1 - bound_ext/2, y2 + bound_ext/2, step)
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

    def show(self, *, show_points=True, bound_ext=10., step=1e-3, fig=None, **plt_kwargs):
        if self._type == ComplexCircle.CIRCLE:
            if show_points:
                for point in self._points:
                    show_complex_point(point, fig, plt_kwargs.get('color', 'b'))
            return _Circle.show(self, show_points=show_points,
                                step=step, fig=fig, **plt_kwargs)
        else:
            return _Line.show(self, show_points=show_points,
                              bound_ext=bound_ext, step=step,
                              fig=fig, **plt_kwargs)

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

    def type(self):
        return self._type


class Region:

    def __init__(self, points):
        self._data = points

    def show(self, fig=None, show_lines=False, bound_ext=10., color='b'):
        for point in self._data:
            show_complex_point(point, fig, color=color)
        if show_lines:
            hor_lines, vert_lines = self.prepare_lines(bound_ext)

            for line in vert_lines + hor_lines:
                line.show(fig=fig, show_points=False,
                          bound_ext=bound_ext, color=color, linestyle='--')

    def prepare_lines(self, bound_ext):
        x_lines = {x.real for x in self._data}
        y_lines = {x.imag for x in self._data}
        vert_lines = [ComplexCircle(complex(x, self._data[0].imag),
                                    complex(x, self._data[0].imag + bound_ext/2),
                                    complex(x, self._data[0].imag - bound_ext/2))
                      for x in x_lines]
        hor_lines = [ComplexCircle(complex(self._data[0].real, y),
                                   complex(self._data[0].real + bound_ext/2, y),
                                   complex(self._data[0].real - bound_ext/2, y))
                     for y in y_lines]
        return hor_lines, vert_lines

    def __iter__(self):
        return iter(self._data)


def region(check_func, bound_real=(-10, 10),
           bound_imag=(-10, 10), step=0.2) -> Region:
    x = np.arange(bound_real[0], bound_real[1], step)
    y = np.arange(bound_imag[0], bound_imag[1], step)
    points = [complex(_x, _y) for _x in x for _y in y if check_func(complex(_x, _y))]
    return Region(points)


class Transformation(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, region: Region):
        pass

    @abstractmethod
    def calc(self, point: complex):
        pass


class FracLinearTransform(Transformation):

    def __init__(self, a: complex, b: complex, c: complex, d: complex):
        """ Fractional linear transformation of complex
        plane W = f(z) = (az + b) / (cz + d).

        Parameters
        ----------
        a, b, c, d - complex numbers - parameters of transformation.
        """
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def calc(self, point: complex):
        if is_inf(point):
            return complex(float('inf'), float('inf')) if self._c == 0 else \
                self._a / self._c
        try:
            res = (self._a * point + self._b) / (self._c * point + self._d)
        except ZeroDivisionError:
            res = complex(float('inf'), float('inf'))
        return res

    def transform(self, region: Region, bound_ext=10.):
        hor_lines, vert_lies = region.prepare_lines(bound_ext)
        res_circles = [self.tcc(x) for x in hor_lines + vert_lies]
        res_points = [self.calc(point) for point in region]
        return Region(res_points), res_circles

    def tcc(self, circle: ComplexCircle) -> ComplexCircle:
        if circle.type() == ComplexCircle.CIRCLE:
            return self._transform_circle(circle._points)
        else:
            return self._transform_line(circle)

    def _transform_circle(self, necc_points) -> ComplexCircle:
        res_points = []
        for point in necc_points:
            res_points.append(self.calc(point))
        return ComplexCircle(*res_points)

    def _transform_line(self, line: _Line) -> ComplexCircle:
        res_points = []
        for point in [line._a, line._b, float('inf')]:
            res_points.append(self.calc(point))
        return ComplexCircle(*res_points)


def test_1():

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


def test_2():
    test_transform = FracLinearTransform(2, 0, 1, -2)
    circle1 = ComplexCircle(complex(2, 0), complex(-2, 0), complex(0, 2))
    circle2 = ComplexCircle(complex(0, 0), complex(2, 0), complex(1, 1))

    circle1.show(color='r', label='Circle 1')
    circle2.show(color='g', label='Circle 2')
    plt.legend()
    plt.show()

    res_circle1 = test_transform.tcc(circle1)
    res_circle2 = test_transform.tcc(circle2)
    res_circle1.show(color='r', label='Image of circle 1')
    res_circle2.show(color='g', label='Image of circle 2')
    plt.legend()
    plt.show()


def test_3():
    test_line = ComplexCircle(0, complex(0, 1), complex(float('inf'), 0))
    test_circle = ComplexCircle(1, -1, complex(0, 1))
    test_line.show(label='test line', color='r')
    test_circle.show(label='test circle', color='g')
    plt.legend()
    plt.show()

    test_transform = FracLinearTransform(1, -1, 1, 1)
    res_test_line = test_transform.tcc(test_line)
    res_test_circle = test_transform.tcc(test_circle)
    res_test_circle.show(label='test circle', color='g')
    res_test_line.show(label='test line', color='r')
    plt.legend()
    plt.show()


def test_4():
    def test_func_check(z: complex):
        return abs(z) < 2 and abs(z - 1) > 1

    test_region = region(test_func_check, bound_real=(-3, 3),
                         bound_imag=(-3, 3), step=0.2)

    test_region.show(show_lines=True, bound_ext=5.)
    plt.show()

    test_transform = FracLinearTransform(2, 0, 1, -2)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    res_region, circles = test_transform.transform(test_region, 5.)
    res_region.show(color='g')
    for el in circles:
        el.show(show_points=False, linestyle='--', color='b')
    plt.show()


if __name__ == "__main__":
    test_4()
