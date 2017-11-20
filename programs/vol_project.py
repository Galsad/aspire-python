#!/opt/anaconda2/bin/python

import argparse

import mrcfile
import numpy as np
from lib.cfft import *
from lib.nufft_cims import py_nufft

import rand_mat


def is_valid_nufft_type(t):
    if (t == "dft" or t == "nufft"):
        return t

    else:
        raise argparse.ArgumentTypeError("type must be nufft or dft")


def is_volume_valid(vol):
    """
    checks if the volume from the user is legal, in case it doesn't, raise
    exception
    :param vol: An array of size n x n x n or a file contains the volume
    :return: In case vol is OK return vol
    """
    if type(vol) == str:
        try:
            vol = mrcfile.open(vol, 'r').data
        except Exception:
            raise argparse.ArgumentTypeError("vol must be an mrc file!")
    if len(vol.shape) != 3 or (
                    vol.shape[0] != vol.shape[1] or vol.shape[0] != vol.shape[
                1]):
        raise argparse.ArgumentTypeError(
            "volume must be  n x n x n - please check your file!")

    return vol


def is_rots_valid(rots):
    """
    checks  whether rots is a valid pickle file or list of valid matrices
    :param rots: rotation matrices or a pickle file containing them
    :return: In Case rots is OK returns rots
    """
    if type(rots) == str:
        try:
            rots = np.array(rand_mat.load_rot_mat(rots))
        except Exception:
            raise argparse.ArgumentTypeError(
                "file must be in a pickle format!")

    if rots.shape[1] != 3 or rots.shape[2] != 3:
        raise argparse.ArgumentTypeError(
            "rotation matrices must be of shape n x 3 x 3!")

    return rots


def choose_nufft_version():
    """
    chooses a good version of nufft from nufft_cims module
    :return: an nufft obj
    """
    # first try fast version of nufft
    nufft_obj = py_nufft.factory('nufft')
    if nufft_obj is None:
        nufft_obj = py_nufft.factory('dft')
        if nufft_obj is None:
            raise "No good version of nufft was found"
    return nufft_obj


def cryo_project(vol, rot_matrices, output_file=None, fft_type=None):
    """
    :param vol: An L-by-L-by-L array containing the voxel structure of a volume
     can be read from a file
    :param rot_matrices: A set of rotation matrices of the form 3-by-3-by-n,
     corresponding to n different
    projection directions - can be read from a file
    :return:An L-by-L-by-n array containing the projections of the volumes in
    the specified directions
    """
    L = vol.shape[0]

    if fft_type is None:
        nufft_obj = choose_nufft_version()

    else:
        nufft_obj = py_nufft.factory(fft_type)

    if L % 2 == 0:
        x, y = np.meshgrid(np.arange(-L / 2, L / 2), np.arange(-L / 2, L / 2))

    n = rot_matrices.shape[0]

    pts_rot = rotated_grids(L, rot_matrices)
    pts_rot = np.transpose(pts_rot.reshape([3, L ** 2 * n], order='F'))

    # this permutation is required for Matlab conventions!
    pts_rot = pts_rot[:, np.argsort([1, 0, 2])]

    im_f = nufft_obj.forward3d(fourier_pts=pts_rot, vol=vol)[0]
    im_f = im_f.reshape([L, L, n], order='F')

    if L % 2 == 0:
        phase_shift = -np.sum(pts_rot, axis=1).reshape([L, L, n],
                                                       order='F') / 2
        for i in range(phase_shift.shape[2]):
            phase_shift[:, :, i] += (2 * np.pi * (x + y + 1) / (2 * L))

        for i in range(phase_shift.shape[2]):
            im_f[:, :, i] *= np.e ** (1j * phase_shift[:, :, i])

    im = icfft2(im_f)

    if L % 2 == 0:
        phase_shift = (2 * np.pi * (x + y) / (2 * L))
        for i in range(im.shape[2]):
            im[:, :, i] *= np.e ** (1j * phase_shift)

    im = np.real(im)

    # permute the dimensions
    im = np.transpose(im, [1, 0, 2])

    if output_file is None:
        return im

    else:
        with mrcfile.new(output_file, overwrite=True) as mrc:
            mrc.set_data(im.astype(np.float32))


def rotated_grids(L, rot_matrices):
    """
    :param L: The resolution of the desired grids.
    :param rot_matrices: An array of size 3-by-3-by-K containing K rotation
    matrices.
    :return: A set of rotated Fourier grids in three dimensions as specified by
     the rotation matrices.
    Frequencies are in the range [-pi, pi].
    """
    grid = np.arange(-L / 2., (L / 2.), dtype=np.float32) / (L / 2.) + 1. / L
    x, y = np.meshgrid(grid, grid)

    num_pts = L ** 2
    num_rots = rot_matrices.shape[0]

    pts = np.pi * np.array([x.flatten(), y.flatten(), np.zeros(num_pts)])
    pts_rot = np.zeros([3, num_pts, num_rots])

    for i in range(num_rots):
        # this line was changed from Matlab - shape
        # of rot_matrices is (k, 3, 3) instead of (3, 3, k)
        pts_rot[:, :, i] = np.dot(rot_matrices[i, :, :], pts)

    pts_rot = pts_rot.reshape([3, L, L, num_rots], order='F')
    return pts_rot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='project a volume according to rotation matrices')
    parser.add_argument("vol", metavar='volume', type=is_volume_valid,
                        help="a path to an mrc file contains the volume")
    parser.add_argument("rotations", metavar='roatiotns', type=is_rots_valid,
                        help="a path to a pickle file contains the rotations")
    parser.add_argument("output", metavar='output', type=str,
                        help="a path to the output file")
    parser.add_argument("-type", metavar='--type', type=is_valid_nufft_type,
                        help="type of nufft can be nufft or dft")

    args = parser.parse_args()
    cryo_project(args.vol, args.rotations, args.output)
