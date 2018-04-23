import benchmark
import numpy as np
from lib.nufft_cims import py_nufft
import lib.nudft_gpu


class benchmark_nufft(benchmark.Benchmark):
    each = 50

    def setUp(self):
        self.size_1d = 1
        self.size_2d = 1000
        self.size_3d = 1
        self.eps = 1.0e-8
        self.type = 'nufft'

    def test_nufft1d_forward(self):
        fourier_pts = np.random.uniform(-np.pi, np.pi, self.size_1d)
        sig = np.random.uniform(-1, 1, self.size_1d) + 1j * np.zeros(
            self.size_1d)

        py_nufft_obj = py_nufft.factory(self.type)
        py_nufft_obj.forward1d(sig, fourier_pts, eps=self.eps)[0]

    # def test_nufft1d_adjoint(self):
    #     fourier_pts = np.random.uniform(-np.pi, np.pi, self.size_1d)
    #     sig_f = np.random.uniform(-1, 1, self.size_1d) + 1j * np.zeros(
    #         self.size_1d)
    #
    #     py_nufft_obj = py_nufft.factory(self.type)
    #     py_nufft_obj.adjoint1d(sig_f, fourier_pts, eps=self.eps)[0]

    def test_nufft2d_forward(self):
        fourier_pts_x = np.random.uniform(-np.pi, np.pi, self.size_2d)
        fourier_pts_y = np.random.uniform(-np.pi, np.pi, self.size_2d)
        fourier_pts = np.transpose(np.array(zip(fourier_pts_x, fourier_pts_y)))
        im = np.random.uniform(-1, 1, self.size_2d * self.size_2d).reshape(
            self.size_2d, self.size_2d)

        py_nufft_obj = py_nufft.factory(self.type)
        py_nufft_obj.forward2d(im, fourier_pts, eps=self.eps)[0]

    # def test_nufft2d_adjoint(self):
    #     fourier_pts_x = np.random.uniform(-np.pi, np.pi, self.size_2d)
    #     fourier_pts_y = np.random.uniform(-np.pi, np.pi, self.size_2d)
    #     fourier_pts = np.array(zip(fourier_pts_x, fourier_pts_y))
    #     im_f = np.random.uniform(-1, 1, self.size_2d) + 1j * np.zeros(
    #         self.size_2d)
    #
    #     py_nufft_obj = py_nufft.factory(self.type)
    #     py_nufft_obj.adjoint2d(im_f, fourier_pts, eps=self.eps)[0]
    #
    # def test_nufft3d_forward(self):
    #     fourier_pts_x = np.random.uniform(-np.pi, np.pi, self.size_3d)
    #     fourier_pts_y = np.random.uniform(-np.pi, np.pi, self.size_3d)
    #     fourier_pts_z = np.random.uniform(-np.pi, np.pi, self.size_3d)
    #     fourier_pts = np.transpose(np.array(
    #         zip(fourier_pts_x, fourier_pts_y, fourier_pts_z)))
    #     vol = np.random.uniform(-1, 1,
    #                             self.size_3d * self.size_3d * self.size_3d).reshape(
    #         self.size_3d, self.size_3d, self.size_3d)
    #
    #     py_nufft_obj = py_nufft.factory(self.type)
    #     py_nufft_obj.forward3d(vol, fourier_pts, eps=self.eps)[0]
    #
    # def test_nufft3d_adjoint(self):
    #     fourier_pts_x = np.random.uniform(-np.pi, np.pi, self.size_3d)
    #     fourier_pts_y = np.random.uniform(-np.pi, np.pi, self.size_3d)
    #     fourier_pts_z = np.random.uniform(-np.pi, np.pi, self.size_3d)
    #     fourier_pts = np.array(
    #         zip(fourier_pts_x, fourier_pts_y, fourier_pts_z))
    #     vol_f = np.random.uniform(-1, 1, self.size_3d) + 1j * np.zeros(
    #         self.size_3d)
    #
    #     py_nufft_obj = py_nufft.factory(self.type)
    #     py_nufft_obj.adjoint3d(vol_f, fourier_pts, eps=self.eps)[0]

    # def test_gpu_nudft1d_forward(self):
    #     fourier_pts = np.random.uniform(-np.pi, np.pi, self.size_1d)
    #     sig = np.random.uniform(-1, 1, self.size_1d) + 1j * np.zeros(
    #         self.size_1d)
    #     pass



'''
class benchmark_nufft_100_double_precision(benchmark_nufft):
    each = 10

    def setUp(self):
        self.size_1d = 100
        self.size_2d = 100
        self.size_3d = 100
        self.eps = 1.0e-16
        self.type = 'nufft'


class benchmark_nufft_256_single_precision(benchmark_nufft):
    each = 10

    def setUp(self):
        self.size_1d = 256
        self.size_2d = 256
        self.size_3d = 256
        self.eps = 1.0e-8
        self.type = 'nufft'


class benchmark_dft_100(benchmark_nufft):
    each = 10

    def setUp(self):
        self.size_1d = 100
        self.size_2d = 100
        self.size_3d = 100
        self.eps = 1.0e-8
        self.type = 'dft'


class benchmark_gpu_dft_100(benchmark_nufft):
    each = 10

    def setUp(self):
        self.size_1d = 100
        self.size_2d = 100
        self.size_3d = 100
        self.eps = 1.0e-8
        self.type = 'gpu_dft'
'''

class benchmark_gpu_nufft_100(benchmark_nufft):
    each = 50

    def setUp(self):
        self.size_1d = 1
        self.size_2d = 1000
        self.size_3d = 1
        self.eps = 1.0e-8
        self.type = 'gpu_nufft'

# class benchmark_nufft_100(benchmark_nufft):
#     each = 50
#
#     def setUp(self):
#         self.size_1d = 1
#         self.size_2d = 512
#         self.size_3d = 1
#         self.eps = 1.0e-8
#         self.type = 'nufft'


if __name__ == '__main__':
    print "Running benchmarks..."
    benchmark.main(format="markdown")
