#!/opt/anaconda2/bin/python
import unittest

import numpy as np
import vol_project


class vol_project_unittest(unittest.TestCase):
    def setUp(self):
        pass

    def test_rotated_grids_identity_odd_size(self, diff=1e-8):
        rot_matrices = np.array([np.eye(3) for i in range(10)])
        res = vol_project.rotated_grids(5, rot_matrices)

        real_res = np.array(
            [[-2.5133, -1.2566, 0, 1.2566, 2.5133],
             [2.5133, 2.5133, 2.5133, 2.5133, 2.5133], [0, 0, 0, 0, 0]])

        self.assertTrue(
            np.sum(np.square(res[:, :, 4, 9] - real_res)) / 15 < diff)

    def test_rotated_grids_30_degrees_odd_size(self, diff=1e-8):
        rot_matrices = np.array(
            [np.array([1, 0, 0, 0, 0.8660, 0.5, 0, -0.5, 0.8660]).reshape(
                [3, 3]) for i in range(10)])
        res = vol_project.rotated_grids(5, rot_matrices)

        real_res = np.array(
            [-2.5133, -1.2566, 0, 1.2566, 2.5133, 2.1765, 2.1765, 2.1765,
             2.1765, 2.1765, -1.2566, -1.2566, -1.2566,
             -1.2566, -1.2566]).reshape([3, 5])

        self.assertTrue(
            np.sum(np.square(res[:, :, 4, 9] - real_res)) / 15 < diff)

    def test_rotated_grids_30_degrees_even_size(self, diff=1e-8):
        rot_matrices = np.array(
            [np.array([1, 0, 0, 0, 0.8660, 0.5, 0, -0.5, 0.8660]).reshape(
                [3, 3]) for i in range(10)])
        res = vol_project.rotated_grids(4, rot_matrices)

        real_res = np.array(
            [-2.3562, -0.7854, 0.7854, 2.3562, 2.0405, 2.0405, 2.0405, 2.0405,
             -1.1781, -1.1781, -1.1781,
             -1.1781]).reshape([3, 4])

        self.assertTrue(
            np.sum(np.square(res[:, :, 3, 9] - real_res)) / 12 < diff)

    def test_vol_project_identity_odd(self, diff=1e-8):
        vol = np.arange(125).reshape([5, 5, 5], order='F')
        rot_matrices = np.array([np.eye(3) for i in range(10)])
        res = vol_project.cryo_project(vol, rot_matrices)

        real_result = np.arange(250, 375, 5).reshape([5, 5])

        self.assertTrue(
            np.sum(np.square(res[:, :, 9] - real_result) / 25) < diff)

    def test_vol_project_30_degrees_odd(self, diff=1e-8):
        vol = np.arange(125).reshape([5, 5, 5], order='F')
        rot_matrices = np.array(
            [np.array([1, 0, 0, 0, 0.8660, 0.5, 0, -0.5, 0.8660]).reshape(
                [3, 3]) for i in range(10)])
        res = vol_project.cryo_project(vol, rot_matrices)

        real_result = np.array(
            [288.3197, 292.4931, 296.6664, 300.8397, 305.0131, 304.1335,
             309.4492, 314.7650, 320.0808, 325.3965, 361.3095, 367.3313,
             373.3531, 379.3749, 385.3968, 333.7579, 339.0736, 344.3894,
             349.7051, 355.0209, 212.4795, 216.6528, 220.8261, 224.9994,
             229.1728]).reshape([5, 5])

        self.assertTrue(
            np.sum(np.square(res[:, :, 9] - real_result)) / 25 < diff)

    def test_vol_project_30_degrees_even(self, diff=1e-8):
        vol = np.arange(64).reshape([4, 4, 4], order='F')
        rot_matrices = np.array(
            [np.array([1, 0, 0, 0, 0.8660, 0.5, 0, -0.5, 0.8660]).reshape(
                [3, 3]) for i in range(10)])
        res = vol_project.cryo_project(vol, rot_matrices)

        real_result = np.array(
            [109.8988, 112.4889, 115.0790, 117.6691, 121.9743, 126.6523,
             131.3304, 136.0085, 158.7095, 163.3876, 168.0656, 172.7437,
             45.5076, 48.0977, 50.6878, 53.2779]).reshape([4, 4])

        self.assertTrue(
            np.sum(np.square(res[:, :, 9] - real_result)) / 16 < diff)
