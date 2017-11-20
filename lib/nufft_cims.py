#!/opt/anaconda2/bin/python

from extern import nufft1df90
from extern import nufft2df90
from extern import nufft3df90

from lib import nudft


class py_nufft():
    def __init__(self):
        pass

    @staticmethod
    def factory(type):
        if type == "nufft": return regular_nufft()
        if type == "dft": return dft()


class regular_nufft(py_nufft):
    @staticmethod
    def adjoint1d(fourier_pts, sig_f, iflag=1, eps=1.0e-8):
        """
        :param fourier_pts: the frequencies in Fourier space at which the
        adjoint Fourier transform is to be
        calculated. These are in the form of a vector of size 1-by-K with
        values in the range [-pi, pi].
        :param sig_f: A Fourier transform calculated at the K frequencies
        specified by fourier_pts. Must be an
        array of size K-by-1.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag >0 for i and <0
        for -i)
        :param eps: determines the precision
        :return: The adjoint Fourier transform of sig_f at frequencies
        fourier_pts.
        """
        sig, err = nufft1df90.nufft1d1f90(fourier_pts, sig_f, iflag, eps,
                                          sig_f.shape[0])
        sig *= sig.shape[0]
        return sig, err

    @staticmethod
    def forward1d(fourier_pts, sig, iflag=-1, eps=1.0e-8):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        Fourier transform is to be calculated.
        These are arranged as an array of size 1-by-K, with values in the
        range [-pi, pi].
        :param sig: An array of size N-by-1 containing a signal.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps: determines the precision
        :return: The Fourier transform of sig at the frequencies fourier_pts
        """
        sig_f, err = nufft1df90.nufft1d2f90(fourier_pts, iflag, eps, sig)
        return sig_f, err

    @staticmethod
    def adjoint2d(fourier_pts, im_f, iflag=1, eps=1.0e-8):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        adjoint Fourier transform is to be calculated.
        These are arranged as a 2-by-K array, with values in the range
        [-pi, pi].
        :param im_f:An image Fourier transform calculated at the frequencies
        specified by fourier_pts.
        This is given as a vector.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps:determines the precision
        :return: The adjoint Fourier transform of im_f at frequencies
        fourier_pts.
        """
        im_size = im_f.shape[0]
        im, err = nufft2df90.nufft2d1f90(fourier_pts[:, 1],
                                         fourier_pts[:, 0], im_f, iflag,
                                         eps, im_size, im_size, im_size)
        im *= im.shape[0]
        return im, err

    @staticmethod
    def forward2d(fourier_pts, im, iflag=-1, eps=1.0e-8):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        Fourier transform is to be calculated.
        These are arranged as a 2-by-K array, with values in the range
         [-pi, pi].
        :param im: An N-by-N array of pixels representing an image.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps: determines the precision
        :return: The Fourier transform of im calculated at the specified
        frequencies
        """
        im_f, err = nufft2df90.nufft2d2f90(fourier_pts[:, 1],
                                           fourier_pts[:, 0], iflag,
                                           eps, im)
        return im_f, err

    @staticmethod
    def adjoint3d(fourier_pts, vol_f, iflag=1, eps=1.0e-8):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        adjoint Fourier transform is to be calculated.
        These are arranged as a 3-by-K array, with values in the range
        [-pi, pi].
        :param vol_f: A volume Fourier transform calculated at the frequencies
        specified by fourier_pts.
        This is given as a vector.
        :param iflag: determines whether i or -i is taken in the fourier
         transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps: determines the precision
        :return: The adjoint Fourier transform of vol_f at frequencies
        fourier_pts
        """

        vol_size = vol_f.shape[0]
        vol, err = nufft3df90.nufft3d1f90(fourier_pts[:, 1],
                                          fourier_pts[:, 0],
                                          fourier_pts[:, 2], vol_f,
                                          iflag, eps, vol_size, vol_size,
                                          vol_size)
        vol *= vol.shape[0]
        return vol, err

    @staticmethod
    def forward3d(fourier_pts, vol, iflag=-1, eps=1.0e-8):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        Fourier transform is to be calculated.
        These are arranged as a 3-by-K array, with values in the range
        [-pi, pi].
        :param vol: An N-by-N-by-N array of voxels representing a volume.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps: determines the precision
        :return:  The Fourier transform of vol calculated at the specified
        frequencies
        """
        vol_f, err = nufft3df90.nufft3d2f90(fourier_pts[:, 1],
                                            fourier_pts[:, 0],
                                            fourier_pts[:, 2],
                                            iflag, eps, vol)
        return vol_f, err


class dft(py_nufft):
    @staticmethod
    def adjoint1d(fourier_pts, sig_f):
        """
        :param fourier_pts: the frequencies in Fourier space at which the
        adjoint Fourier transform is to be
        calculated. These are in the form of a vector of size 1-by-K with
        values in the range [-pi, pi].
        :param sig_f: A Fourier transform calculated at the K frequencies
        specified by fourier_pts. Must be an
        array of size K-by-1.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag >0 for i and <0
        for -i)
        :param eps: determines the precision
        :return: The adjoint Fourier transform of sig_f at frequencies
        fourier_pts.
        """

        sig = nudft.anudft1(sig_f, fourier_pts, sig_f.shape[0])
        return sig, 0

    @staticmethod
    def forward1d(fourier_pts, sig):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        Fourier transform is to be calculated.
        These are arranged as an array of size 1-by-K, with values in the
        range [-pi, pi].
        :param sig: An array of size N-by-1 containing a signal.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps: determines the precision
        :return: The Fourier transform of sig at the frequencies fourier_pts
        """
        sig_f = nudft.nudft1(sig, fourier_pts)
        return sig_f, 0

    @staticmethod
    def adjoint2d(fourier_pts, im_f):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        adjoint Fourier transform is to be calculated.
        These are arranged as a 2-by-K array, with values in the range
        [-pi, pi].
        :param im_f:An image Fourier transform calculated at the frequencies
        specified by fourier_pts.
        This is given as a vector.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps:determines the precision
        :return: The adjoint Fourier transform of im_f at frequencies
        fourier_pts.
        """

        sig = nudft.anudft2(im_f, fourier_pts, im_f.shape[0])
        return sig, 0

    @staticmethod
    def forward2d(fourier_pts, im):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        Fourier transform is to be calculated.
        These are arranged as a 2-by-K array, with values in the range
         [-pi, pi].
        :param im: An N-by-N array of pixels representing an image.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps: determines the precision
        :return: The Fourier transform of im calculated at the specified
        frequencies
        """
        im_f = nudft.nudft2(im, fourier_pts)
        return im_f, 0

    @staticmethod
    def adjoint3d(fourier_pts, vol_f):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        adjoint Fourier transform is to be calculated.
        These are arranged as a 3-by-K array, with values in the range
        [-pi, pi].
        :param vol_f: A volume Fourier transform calculated at the frequencies
        specified by fourier_pts.
        This is given as a vector.
        :param iflag: determines whether i or -i is taken in the fourier
         transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps: determines the precision
        :return: The adjoint Fourier transform of vol_f at frequencies
        fourier_pts
        """

        sig = nudft.anudft3(vol_f, fourier_pts, vol_f.shape[0])
        return sig, 0

    @staticmethod
    def forward3d(fourier_pts, vol):
        """
        :param fourier_pts: The frequencies in Fourier space at which the
        Fourier transform is to be calculated.
        These are arranged as a 3-by-K array, with values in the range
        [-pi, pi].
        :param vol: An N-by-N-by-N array of voxels representing a volume.
        :param iflag: determines whether i or -i is taken in the fourier
        transform formula (iflag > 0 for i and < 0
        for -i)
        :param eps: determines the precision
        :return:  The Fourier transform of vol calculated at the specified
        frequencies
        """
        vol_f = nudft.nudft3(vol, fourier_pts)
        return vol_f, 0


if __name__ == "__main__":
    pass
