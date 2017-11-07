import logging
import numpy
import nufft1df90
import nufft2df90
import nufft3df90

print nufft2df90.__file__

print (nufft2df90.nufft2d2f90.__doc__)

n=5
iflag=1
eps=1.0e-16

sig_f=numpy.linspace(-1,1,n)+1j*numpy.zeros(n)
sig_f=numpy.multiply(sig_f,sig_f)

fourier_pts=numpy.linspace(-numpy.pi/2.0, numpy.pi, n)

sig, ier = nufft1df90.nufft1d1f90(fourier_pts, sig_f, iflag, eps, n)
sig = sig*5

print "sig_f: ", sig_f

print "fourier_pts: ", fourier_pts

print "sig: ", sig

print "ier: ", ier


n=5
iflag=1
eps=1.0e-16

sig_f=numpy.linspace(-1,1,n)+1j*numpy.zeros(n)
sig_f=numpy.multiply(sig_f,sig_f)

fourier_pts=numpy.linspace(-numpy.pi/2.0, numpy.pi, n)

sig, ier = nufft1df90.nufft1d2f90(fourier_pts, iflag, eps, sig_f)

print "sig_f: ", sig_f

print "fourier_pts: ", fourier_pts

print "sig: ", sig

print "ier: ", ier

fourier_pts_x = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)
fourier_pts_y = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)
fourier_pts = numpy.array(zip(fourier_pts_x, fourier_pts_y))
sig_f = numpy.linspace(-numpy.pi/2, numpy.pi, 5)

# im = numpy.arange(n*n).reshape(n, n) - (n*n/2)
# N = im.shape[0]

# grid = numpy.arange(numpy.ceil(-N / 2.), numpy.ceil(N / 2.)).astype(numpy.complex64)
# grid_x, grid_y = numpy.meshgrid(grid, grid) # grid_y and grid_x are like matlab convensions
#
# pts = numpy.array([grid_x.flatten(), grid_y.flatten()]).astype(numpy.complex64)

sig, ier = nufft2df90.nufft2d1f90(fourier_pts_x, fourier_pts_y, sig_f, iflag, eps, 5, 5)

sig = sig*5

print "sig_f: ", sig_f

print "fourier_pts: ", fourier_pts

print "sig: ", sig

print "ier: ", ier


fourier_pts_x = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)
fourier_pts_y = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)
fourier_pts = numpy.array(zip(fourier_pts_x, fourier_pts_y))
# sig_f = numpy.linspace(-numpy.pi/2, numpy.pi, 5)

n=5
eps=1.0e-16
im = numpy.arange(n*n).reshape(n, n) - (n*n/2)
#converting to complex number
#im = im.astype(numpy.complex128)
im = numpy.ascontiguousarray(im, dtype=numpy.complex128)
N = im.shape[0]

# grid = numpy.arange(numpy.ceil(-N / 2.), numpy.ceil(N / 2.)).astype(numpy.complex64)
# grid_x, grid_y = numpy.meshgrid(grid, grid) # grid_y and grid_x are like matlab convensions
#
# pts = numpy.array([grid_x.flatten(), grid_y.flatten()]).astype(numpy.complex64)
print "im: ", im

sig, ier = nufft2df90.nufft2d2f90(fourier_pts_x, fourier_pts_y, -iflag, eps, im)

print "sig: ", sig

print "fourier_pts: ", fourier_pts

print "ier: ", ier

fourier_pts_x = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)
fourier_pts_y = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)
fourier_pts_z = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)

sig = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)
iflag=1
eps=1.0e-16

fourier_pts = numpy.array(zip(fourier_pts_x, fourier_pts_y, fourier_pts_z))

vol_f, ier = nufft3df90.nufft3d1f90(fourier_pts_x, fourier_pts_y, fourier_pts_z, sig, iflag, eps, 5, 5, 5)

print "vol_f: ", vol_f
print "ier: ", ier


fourier_pts_x = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)
fourier_pts_y = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)
fourier_pts_z = numpy.linspace(-numpy.pi/2.0,numpy.pi,n)

vol_f = numpy.arange(125).reshape((5, 5, 5))
iflag=1
eps=1.0e-16

vol, ier = nufft3df90.nufft3d2f90(fourier_pts_x, fourier_pts_y, fourier_pts_z, iflag, eps, vol_f, 5, 5, 5)

print "cj: ", vol
print "ier: ", ier
