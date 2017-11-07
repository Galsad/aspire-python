import numpy

def nudft1(sig, fourier_pts):
    '''
    python version of forward nudft 1d
    :param sig:
    :param fourier_pts:
    :return:
    '''
    fourier_pts = fourier_pts.astype(numpy.complex64)
    sig = sig.astype(numpy.complex64)

    sig_f = numpy.zeros(len(fourier_pts))
    sig_f = sig_f.astype(numpy.complex64)

    grid = numpy.arange(numpy.ceil(-len(sig)/2.), numpy.ceil(len(sig)/2.)).astype(numpy.complex64)

    for i in range(len(fourier_pts)):
        sig_f[i] = numpy.dot(numpy.e**(0-1j * fourier_pts[i]*grid), sig)
    return sig_f


def anudft1(sig_f, fourier_pts, sz):
    '''
    python version of adjoint nudft1
    :param sig_f:
    :param fourier_pts:
    :param sz:
    :return:
    '''
    grid = numpy.arange(numpy.ceil(-len(sig_f) / 2.), numpy.ceil(len(sig_f) / 2.)).astype(numpy.complex64)
    sig = numpy.zeros(sz).astype(numpy.complex64)
    fourier_pts = fourier_pts.astype(numpy.complex64)
    sig_f = sig_f.astype(numpy.complex64)

    for i in range(len(fourier_pts)):
        sig[i] = numpy.dot(numpy.e**(0+1j * fourier_pts*grid[i]), sig_f)
    return sig

def nudft2(im, fourier_pts):
    '''
    :param im:
    :param fourier_pts:
    :return:
    '''
    assert im.shape[0] == im.shape[1]
    im = im.astype(numpy.complex64)
    N = im.shape[0]

    im_f = numpy.zeros(N).astype(numpy.complex64)
    grid = numpy.arange(numpy.ceil(-N / 2.), numpy.ceil(N / 2.)).astype(numpy.complex64)
    grid_x, grid_y = numpy.meshgrid(grid, grid) # grid_y and grid_x are like matlab convensions

    pts = numpy.array([grid_x.flatten(), grid_y.flatten()]).astype(numpy.complex64)

    for i in range(fourier_pts.shape[0]):
        im_f[i] = numpy.dot(numpy.e**(0-1j * numpy.dot(fourier_pts[i, :], pts)), im.flatten())

    return im_f

def anudft2(im_f, fourier_pts, sz):
    '''
    :param im:
    :param fourier_pts:
    :return:
    '''
    im_f = im_f.astype(numpy.complex64)
    N = sz

    im = numpy.zeros(N * N).astype(numpy.complex64)
    grid = numpy.arange(numpy.ceil(-N / 2.), numpy.ceil(N / 2.)).astype(numpy.complex64)
    grid_x, grid_y = numpy.meshgrid(grid, grid) # grid_y and grid_x are like matlab convensions

    pts = numpy.array([grid_x.flatten(), grid_y.flatten()]).astype(numpy.complex64)

    for i in range(pts.shape[1]):
        im[i] = numpy.dot(numpy.e**(0+1j * numpy.dot(fourier_pts, pts[:, i])), im_f.flatten())

    return im.reshape((N, N))

def nudft3(vol, fourier_pts):
    '''
    :param vol:
    :param fourier_pts:
    :return:
    '''
    assert vol.shape[0] == vol.shape[1]
    assert vol.shape[0] == vol.shape[2]
    vol = vol.astype(numpy.complex64)
    N = vol.shape[0]

    vol_f = numpy.zeros(N).astype(numpy.complex64)
    grid = numpy.arange(numpy.ceil(-N / 2.), numpy.ceil(N / 2.)).astype(numpy.complex64)
    grid_x, grid_y, grid_z = numpy.meshgrid(grid, grid, grid) # grid_y and grid_x are like matlab convensions

    pts = numpy.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).astype(numpy.complex64)

    for i in range(fourier_pts.shape[0]):
        vol_f[i] = numpy.dot(numpy.e**(0-1j * numpy.dot(fourier_pts[i, :], pts)), vol.flatten())

    return vol_f

def anudft3(vol_f, fourier_pts, sz):
    '''
    :param im:
    :param fourier_pts:
    :return:
    '''
    vol_f = vol_f.astype(numpy.complex64)
    N = sz

    vol = numpy.zeros(N * N * N).astype(numpy.complex64)
    grid = numpy.arange(numpy.ceil(-N / 2.), numpy.ceil(N / 2.)).astype(numpy.complex64)
    grid_x, grid_y, grid_z = numpy.meshgrid(grid, grid, grid) # grid_y and grid_x are like matlab convensions

    pts = numpy.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).astype(numpy.complex64)

    for i in range(pts.shape[1]):
        vol[i] = numpy.dot(numpy.e**(0+1j * numpy.dot(fourier_pts, pts[:, i])), vol_f.flatten())

    return vol.reshape((N, N, N))