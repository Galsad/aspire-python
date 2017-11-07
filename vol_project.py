#!/opt/anaconda2/bin/python
import numpy as np
from py_nufft import py_nufft
from uniform_fft_utils import *
import mrcfile
import rand_mat
import sys


def cryo_project(vol, rot_matrices, output_file=None):
    """
    :param vol: An L-by-L-by-L array containing the voxel structure of a volume can be read from a file
    :param rot_matrices: A set of rotation matrices of the form 3-by-3-by-n, corresponding to n different
    projection directions - can be read from a file
    :return:An L-by-L-by-n array containing the projections of the volumes in the specified directions
    """
    if type(vol) == str and type(rot_matrices) == str and output_file != None:
        try:
            vol = mrcfile.open(vol, 'r').data
            rot_matrices = np.array(rand_mat.load_rot_mat(rot_matrices))
            print rot_matrices
            assert(type(output_file) == str)
        except:
            raise "Usage: vol - mrc file path, rot_matrices - pickle file, outputfile - mrc file path"

    L = vol.shape[0]

    # Volume must has 3 dims
    assert len(vol.shape) == 3

    # volume must be of size LxLxL
    assert vol.shape[0] == vol.shape[1] and vol.shape[0] == vol.shape[2]

    if L % 2 == 0:
        x, y = np.meshgrid(np.arange(-L / 2, L / 2), np.arange(-L / 2, L / 2))

    n = rot_matrices.shape[0]

    pts_rot = rotated_grids(L, rot_matrices)
    pts_rot = np.transpose(pts_rot.reshape([3, L ** 2 * n], order='F'))

    # this permutation is required for Matlab conventions!
    pts_rot = pts_rot[:, np.argsort([1, 0, 2])]

    im_f = py_nufft.forward3d(fourier_pts=pts_rot, vol=vol)[0]
    im_f = im_f.reshape([L, L, n], order='F')

    if L % 2 == 0:
        phase_shift = -np.sum(pts_rot, axis=1).reshape([L, L, n], order='F') / 2
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

    if output_file == None:
        return im

    else:
        with mrcfile.new(output_file, overwrite=True) as mrc:
            mrc.set_data(im.astype(np.float32))

def rotated_grids(L, rot_matrices):
    """
    :param L: The resolution of the desired grids.
    :param rot_matrices: An array of size 3-by-3-by-K containing K rotation matrices.
    :return: A set of rotated Fourier grids in three dimensions as specified by the rotation matrices.
    Frequencies are in the range [-pi, pi].
    """
    grid = np.arange(-L / 2., (L / 2.), dtype=np.float32) / (L / 2.) + 1. / L
    x, y = np.meshgrid(grid, grid)

    num_pts = L ** 2
    num_rots = rot_matrices.shape[0]

    pts = np.pi * np.array([x.flatten(), y.flatten(), np.zeros(num_pts)])
    pts_rot = np.zeros([3, num_pts, num_rots])

    for i in range(num_rots):
        # this line was changed from Matlab - shape of rot_matrices is (k, 3, 3) instead of (3, 3, k)
        pts_rot[:, :, i] = np.dot(rot_matrices[i, :, :], pts)

    pts_rot = pts_rot.reshape([3, L, L, num_rots], order='F')
    return pts_rot


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise "Usage: vol - mrc file path, rot_matrices - pickle file, outputfile - mrc file path"
    vol = sys.argv[1]
    rot_matrices = sys.argv[2]
    output_file = sys.argv[3]
    cryo_project(vol, rot_matrices, output_file)