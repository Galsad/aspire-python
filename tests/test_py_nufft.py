#!/opt/anaconda2/bin/python
import logging
import sys
import unittest

import nudft
import numpy as np
from nufft_cims import py_nufft

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class nufft_unitest(unittest.TestCase):
    def setUp(self):
        pass

    def test_nufft_forward_1d(self, n=34, diff=1e-8):
        '''
        :param n: the size of the fft
        comapres the python nudft1_d results to the fortran run
        generate fourier points in a constant way
        :return:
        '''
        fourier_pts = np.random.uniform(-np.pi, np.pi, n)
        sig = np.random.uniform(-1, 1, n) + 1j * np.zeros(n)
        fortran_results = py_nufft.forward1d(fourier_pts, sig)[0]
        python_results = nudft.nudft1(sig, fourier_pts)

        self.assertTrue(np.sum(
            np.square(np.abs(python_results - fortran_results))) / n < diff)

    def test_nufft_adjoint_1d(self, n=34, diff=1e-8):
        '''
        :param n: the size of the fft
        comapres the python nudft1_d results to the fortran run
        generate fourier points in a constant way
        :return:
        '''
        fourier_pts = np.random.uniform(-np.pi, np.pi, n)
        sig_f = np.random.uniform(-1, 1, n) + 1j * np.zeros(n)
        fortran_results = py_nufft.adjoint1d(fourier_pts, sig_f)[0]
        python_results = nudft.anudft1(sig_f, fourier_pts, n)

        self.assertTrue(np.sum(
            np.square(np.abs(python_results - fortran_results))) / n < diff)

    def test_nufft_forward_2d(self, n=34, diff=1e-8):
        '''
        :param n: the size of the fft
        comapres the python nudft2_d results to the fortran run
        generate fourier points in a constant way
        :return:
        '''
        fourier_pts_x = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts_y = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts = np.array(zip(fourier_pts_x, fourier_pts_y))
        im = np.random.uniform(-1, 1, n * n).reshape(n, n)
        fortran_results = py_nufft.forward2d(fourier_pts, im)[0]
        python_results = nudft.nudft2(im, fourier_pts)

        self.assertTrue(
            np.sum(np.square(np.abs(python_results - fortran_results))) / (
                n * n) < diff)

    def test_nufft_adjoint_2d(self, n=34, diff=1e-8):
        '''
        :param n: the size of the fft
        comapres the python nudft2_d results to the fortran run
        generate fourier points in a constant way
        :return:
        '''
        fourier_pts_x = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts_y = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts = np.array(zip(fourier_pts_x, fourier_pts_y))
        im_f = np.random.uniform(-1, 1, n) + 1j * np.zeros(n)
        fortran_results = py_nufft.adjoint2d(fourier_pts, im_f)[0]
        python_results = nudft.anudft2(im_f, fourier_pts, n)

        self.assertTrue(np.sum(
            np.square(np.abs(python_results - fortran_results))) / n < diff)

    def test_nufft_forward_3d(self, n=34, diff=1e-8):
        '''
        :param n: the size of the fft
        comapres the python nudft3_d results to the fortran run
        generate fourier points in a constant way
        :return:
        '''
        fourier_pts_x = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts_y = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts_z = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts = np.array(
            zip(fourier_pts_x, fourier_pts_y, fourier_pts_z))
        vol = np.random.uniform(-1, 1, n * n * n).reshape(n, n, n)

        fortran_results = py_nufft.forward3d(fourier_pts, vol)[0]
        python_results = nudft.nudft3(vol, fourier_pts)

        self.assertTrue(
            np.sum(np.square(np.abs(python_results - fortran_results))) / (
                n * n * n) < diff)

    def test_nufft_adjoint_3d(self, n=34, diff=1e-8):
        '''
        :param n: the size of the fft
        comapres the python nudft3_d results to the fortran run
        generate fourier points in a constant way
        :return:
        '''

        fourier_pts_x = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts_y = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts_z = np.random.uniform(-np.pi, np.pi, n)
        fourier_pts = np.array(
            zip(fourier_pts_x, fourier_pts_y, fourier_pts_z))
        vol_f = np.random.uniform(-1, 1, n) + 1j * np.zeros(n)

        fortran_results = py_nufft.adjoint3d(fourier_pts, vol_f)[0]
        python_results = nudft.anudft3(vol_f, fourier_pts, n)

        self.assertTrue(np.sum(
            np.square(np.abs(python_results - fortran_results))) / n < diff)


if __name__ == '__main__':
    unittest.main(verbosity=True)
