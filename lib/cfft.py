import numpy as np

# The FFT is computed using O(nlogn) operations.
# x   The sequence whose FFT should be computed.
#     Can be of odd or even length. Must be a 1-D vector.
#
# Returns the aliased FFT of the sequence x.
cfft = lambda x: np.fft.ifftshift(np.fft.fft(np.fft.fftshift(x)))

# Aliased n-dimensional Inverse FFT of the array x.
# The inverse FFT is computed using O((n^d)logn) operations,
# where d is the dimension of the image.
#
# x    The frequency image whose inverse FFT should be computed.
#      Can be of odd or even length in each dimension.
# Returns the aliased n-dimensional inverse FFT of the array x.
#
icfft2 = lambda x: np.fft.fftshift(np.transpose(np.fft.ifft2(np.transpose(np.fft.ifftshift(x)))))
