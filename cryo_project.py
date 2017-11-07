import numpy as np
import py_nufft
from tqdm import tqdm
import numpy.matlib
import scipy.io


def cryo_project(vol, rot_matrices, half_pixel=False):
    L = vol.shape[0]

    # Volume must has 3 dims
    assert len(vol) == 3

    # volume must be of size LxLxL
    assert vol.shape[0] == vol.shape[1] and vol.shape[0] == vol.shape[2]

    if half_pixel == True and L % 2 == 0:
        X, Y = np.meshgrid(range(-L / 2, L / 2), range(-L / 2, L / 2))

    # n is the amount of rotate matrices
    n = rot_matrices.shape[2]

    # TODO - need to reshape pts_rot
    pst_rot = rotated_grids(L, rot_matrices, half_pixel)
    # pts_rot = reshape(pts_rot, )

    # TODO - need to verify parameters
    im_f = py_nufft.py_nufft.forward3d(fourier_pts=pst_rot, vol=vol, iflag=1)
    # im_f = reshape(im_f, )


# TODO
def rotated_grids(L, rot_matrices, half_pixel=False):
    """
    :param L: The resolution of the desired grids.
    :param rot_matrices: An array of size 3-by-3-by-K containing K rotation matrices.
    :param half_pixel: If true, centers the rotation around a half-pixel
    :return: A set of rotated Fourier grids in three dimensions as specified by the rotation matrices.
    Frequencies are in the range [-pi, pi].
    """

    # TODO -- matlab meshgrid has also phi and theta - need to add them in python (?)
    grid = np.ceil(np.array(-L / 2, (L / 2) + 1) / (L / 2))
    x, y = np.meshgrid(grid, grid)

    if L%2 == 0 and half_pixel:
        x += 1./L
        y += 1. / L

    num_pts = L ** 2
    num_rots = rot_matrices.shape[2]

    pts = np.pi * [x, y, np.zeros(num_pts, 1)]
    pts_rot = np.zeros(3, num_pts, num_rots)

    for i in range(num_rots):
        pts_rot[:, :, i] = rot_matrices[:, :, i] * pts

    pts_rot = pts_rot.reshape([3, L, L, num_rots])
    return pts_rot


def mesh_2d(L, inclusive=False):
    """
    :param L: The dimension of the desired mesh
    :param inclusive: Specifies whether both endpoints -1 and +1 should be included in each axis. If true, -1 and +1
     are always included, and so 0 is only included for odd L. If true, -1 is included for even L, while for odd L,
     neither -1 or +1 is included
    :return:
    """

    # TODO -- need to figure out how to use p_meshes
    if (not inclusive):
        grid = np.ceil(np.array(-L / 2, (L / 2) + 1) / (L / 2))
    else:
        grid = np.ceil(np.array(-(L - 1) / 2, (L + 1) / 2) / ((L - 1) / 2))


# TODO
# need to make sure use can use this function with less than 4 params
# def cryo_project(vol, rot, n, precision):
#     if (n % 2 == 1):
#         grid = np.arange(-(n-1)/2, (n-1)/2 + 1)
#     else:
#         grid = np.arange(-n/2, n/2) + 0.5
#
#     grid_x, grid_y = np.meshgrid(grid, grid)
#
#     nv = vol.shape[0]
#
#     if n > nv:
#         if ((n-nv)%2 == 1):
#             raise "Upsampling from odd to even sizes or vice versa is currently not supported"
#         dn = np.floor((n-nv)/2) # why floor is needed?
#         fv = cfft(vol)
#         padded_volume = np.zeros([n, n, n])
#         padded_volume[dn + 1:dn + nv, dn + 1:dn + nv, dn + 1:dn + nv] = fv
#         vol = icfftn(padded_volume)
#         nv=n
#
#     K = rot.shape[2]
#     projections = np.zeros([n, n, K])
#
#
#     for k in tqdm(range(K)):
#         R = rot[:, :, k]
#         Rt = np.transpose(R)
#
#         n_x = Rt[:, 0]
#         n_y = Rt[:, 1]
#
#         P = -2. * np.pi * (grid_x.reshape(n*n, 1) * n_x.reshape(1, 3) + grid_y.reshape(n*n, 1) * n_y.reshape(1, 3))/nv
#
#         column_permutation = np.argsort([1, 0, 2])
#         P = P[:, column_permutation]
#
#
#         nufft3 = py_nufft.py_nufft3d(-np.conjugate(P), iflag=1, eps=precision)
#         nufft3.set_vol(vol)
#         nufft3.forward()
#
#         print "P is: ", P
#
#         projection_fourier = nufft3.vol_f
#         print "nufft3.volf is: ", nufft3.vol_f
#
#         if n%2 == 0:
#             projection_fourier *= np.e ** (1.0j * np.sum(P, axis=1) / 2.)
#             projection_fourier *= (np.e ** (1.0j * 2 * np.pi * (grid_x.flatten() + grid_y.flatten() - 1) / (2*n)))
#
#         projection_fourier = projection_fourier.reshape(n, n)
#         projection_fourier = np.fft.ifftshift(projection_fourier)
#         projection = np.fft.fftshift(np.fft.ifft2(projection_fourier))
#
#         if n%2 == 0:
#             projection *= (np.e ** (1.0j * 2 * np.pi * (grid_x.flatten() + grid_y.flatten()) / (2*n))).reshape([n, n])
#
#         projections[:, :, k] = np.real(projection)
#     return projections


if __name__ == "__main__":
    vol_mat = scipy.io.loadmat(r'../../aspire/projections/simulation/vol.mat')['volume']

    for_mat = scipy.io.loadmat(r'../../aspire/projections/simulation/proj.mat')['projection_fourier'].reshape(1, 36)

    # vol = np.random.uniform(-np.pi, np.pi, 125).reshape([5, 5, 5])
    vol = np.arange(1, 126).reshape([5, 5, 5])
    vol = np.transpose(vol)

    # vol = np.reshape(np.ones(125), [5, 5, 5])
    rot = np.matlib.repmat(np.eye(3), 1, 10).reshape([3, 3, 10])
    cryo_project(vol, rot, 6, 1e-16)
