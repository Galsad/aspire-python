import numpy as np

def nudft1(sig, fourier_pts):
    """
    Non-uniform discrete Fourier transform (1D)
    :param sig: An array of size n containing the signal
    :param fourier_pts: An array of size K represents the frequencies in
    Fourier space at which the Fourier transform
    is to be calculated.
    :return: An Array of size K represents the Fourier transform of sig
    calculated at the specified frequencies.
    """
    fourier_pts = fourier_pts.astype(np.complex64)
    sig = sig.astype(np.complex64)

    sig_f = np.zeros(len(fourier_pts))
    sig_f = sig_f.astype(np.complex64)

    grid = np.arange(np.ceil(-len(sig) / 2.), np.ceil(len(sig) / 2.)).astype(
        np.complex64)

    for i in range(len(fourier_pts)):
        sig_f[i] = np.dot(np.e ** (0 - 1j * fourier_pts[i] * grid), sig)
    return sig_f


def anudft1(sig_f, fourier_pts, sz):
    """
    Adjoint Non-uniform discrete Fourier transform (1D)
    :param sig_f: A Fourier transform calculated at the K frequencies
    specified by fourier_pts
    :param fourier_pts: fourier_pts: An array of size K represents the
    frequencies in Fourier space at which the Fourier
    transform is to be calculated.
    :param sz: signal size
    :return: An array of size (sz) represents the adjoint Fourier transform
    of sig_f at frequencies fourier_pts.
    """
    grid = np.arange(np.ceil(-sz / 2.),
                     np.ceil(sz / 2.)).astype(np.complex64)
    sig = np.zeros(sz).astype(np.complex64)
    fourier_pts = fourier_pts.astype(np.complex64)
    sig_f = sig_f.astype(np.complex64)

    for i in range(len(fourier_pts)):
        sig[i] = np.dot(np.e ** (0 + 1j * fourier_pts * grid[i]), sig_f)
    return sig


def nudft2(im, fourier_pts):
    """
    Non-uniform discrete Fourier transform (2D)
    :param im: An array of size nxn containing the image
    :param fourier_pts: An array of size K represents the frequencies in
    Fourier space at which the Fourier transform
    is to be calculated.
    :return: An Array of size K represents the Fourier transform of sig
    calculated at the specified frequencies.
    """
    assert im.shape[0] == im.shape[1]
    im = im.astype(np.complex64)
    N = im.shape[0]

    im_f = np.zeros(N).astype(np.complex64)
    grid = np.arange(np.ceil(-N / 2.), np.ceil(N / 2.)).astype(np.complex64)
    # grid_y and grid_x are like matlab conventions
    grid_x, grid_y = np.meshgrid(grid,
                                 grid)

    pts = np.array([grid_x.flatten(), grid_y.flatten()]).astype(np.complex64)


    for i in range(fourier_pts.shape[0]):
        im_f[i] = np.dot(np.e ** (0 - 1j * np.dot(fourier_pts[i, :], pts)),
                         im.flatten())

    return im_f


def anudft2(im_f, fourier_pts, sz):
    """
    Adjoint Non-uniform discrete Fourier transform (2D)
    :param im_f: A Fourier transform calculated at the K frequencies specified
    by fourier_pts
    :param fourier_pts: fourier_pts: An array of size K represents the
    frequencies in Fourier space at which the Fourier
    transform is to be calculated.
    :param sz: image size
    :return: An array of size (sz X sz) represents the adjoint Fourier
    transform of im_f at frequencies fourier_pts.
    """
    im_f = im_f.astype(np.complex64)
    N = sz

    im = np.zeros(N * N).astype(np.complex64)
    grid = np.arange(np.ceil(-N / 2.), np.ceil(N / 2.)).astype(np.complex64)
    # grid_y and grid_x are like matlab convensions
    grid_x, grid_y = np.meshgrid(grid, grid)

    pts = np.array([grid_x.flatten(), grid_y.flatten()]).astype(np.complex64)

    for i in range(pts.shape[1]):
        im[i] = np.dot(np.e ** (0 + 1j * np.dot(fourier_pts, pts[:, i])),
                       im_f.flatten())

    return im.reshape((N, N))


def nudft3(vol, fourier_pts):
    """
    Non-uniform discrete Fourier transform (3D)
    :param vol: An array of size n containing the volume
    :param fourier_pts: An array of size K represents the frequencies in
    Fourier space at which the Fourier transform
    is to be calculated.
    :return: An Array of size K represents the Fourier transform of sig
    calculated at the specified frequencies.
    """
    assert vol.shape[0] == vol.shape[1]
    assert vol.shape[0] == vol.shape[2]
    vol = vol.astype(np.complex64)
    N = vol.shape[0]

    vol_f = np.zeros(N).astype(np.complex64)
    grid = np.arange(np.ceil(-N / 2.), np.ceil(N / 2.)).astype(np.complex64)
    # grid_y and grid_x are like matlab convensions
    grid_x, grid_y, grid_z = np.meshgrid(grid, grid,
                                         grid)

    pts = np.array(
        [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).astype(
        np.complex64)

    for i in range(fourier_pts.shape[0]):
        vol_f[i] = np.dot(np.e ** (0 - 1j * np.dot(fourier_pts[i, :], pts)),
                          vol.flatten())

    return vol_f


def anudft3(vol_f, fourier_pts, sz):
    """
    Adjoint Non-uniform discrete Fourier transform (3D)
    :param vol_f: A Fourier transform calculated at the K frequencies
    specified by fourier_pts
    :param fourier_pts: fourier_pts: An array of size K represents the
    frequencies in Fourier space at which the Fourier
    transform is to be calculated.
    :param sz: image size
    :return: An array of size (sz X sz X sz) represents the adjoint Fourier
    transform of im_vol at frequencies fourier_pts.
    """
    vol_f = vol_f.astype(np.complex64)
    N = sz

    vol = np.zeros(N * N * N).astype(np.complex64)
    grid = np.arange(np.ceil(-N / 2.), np.ceil(N / 2.)).astype(np.complex64)
    # grid_y and grid_x are like matlab convensions
    grid_x, grid_y, grid_z = np.meshgrid(grid, grid,
                                         grid)

    pts = np.array(
        [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).astype(
        np.complex64)

    for i in range(pts.shape[1]):
        vol[i] = np.dot(np.e ** (0 + 1j * np.dot(fourier_pts, pts[:, i])),
                        vol_f.flatten())

    return vol.reshape((N, N, N))
