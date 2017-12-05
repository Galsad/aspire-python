import pycuda.driver as cuda
#import must stay here even if it's not used directly!
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.gpuarray as gpuarray
from lib import nudft
import nufft_cims


def calculate_kernel(omega, b, m, q):
    n = len(alpha)
    mu = np.round(omega * m)
    P = np.zeros([n, q + 1])

    # creating kernel function
    for i in range(-q / 2, q / 2 + 1):
        P[:, i + q / 2] = (np.exp(
            -(m * omega - (mu + i)) ** 2 / (4 * b))) / (
                              2 * np.sqrt(b * np.pi))

    return P

class nufft_gpu():


    # TODO - fix problem with dimension of kernel function  
    @staticmethod
    def forward1d(sig, fourier_pts, eps=None):

        # preparing kernel #
        b = 0.5993
        m = 2
        q = 10

        ker = calculate_kernel(sig, b, m, q)

        n = len(fourier_pts)

        # TODO - verify it
        M = n
        mu = np.round(sig * m).astype(np.int32)
        tau = np.zeros(M * m).astype(np.complex64)
        offset = np.ceil((m * M - 1) / 2.)
        # ker = ker.astype(np.complex64)

        # converting all variables to complex one
        sig = sig.astype(np.complex64)
        fourier_pts = fourier_pts.astype(np.complex64)

        # the GPU kernel
        mod = SourceModule("""
        #include <pycuda-complex.hpp>
        #include <stdio.h>
        __global__ void nufft1(pycuda::complex<float> *fourier_pts, pycuda::complex<float> **ker, int *mu, int offset, int q, int M, int m, pycuda::complex<float> *tau)
        {
            //printf("HI!\\n");
            int idx = (threadIdx.x + blockDim.x * blockIdx.x) + 
            (threadIdx.y + blockDim.y * blockIdx.y) + 
            (threadIdx.z + blockDim.z * blockIdx.z);
            
            
            // initialize grid
            int i = -q / 2;
            int inner_idx = 0;
            for (i = -q/2; i<q/2; i++){
                printf("number is %d %d \\n", idx, mu[idx]);
                inner_idx = mu[idx] + i;
                inner_idx = (inner_idx + offset + m*M) % (M * m);
                printf("inner index is: %d\\n", inner_idx);
                tau[inner_idx] = tau[inner_idx] + ker[idx][i + q/2] * fourier_pts[idx];
            }
        }
        """)

        bdim = (len(mu), 1, 1)

        gridm = (n / bdim[0] + (n % bdim[0] > 0), 1)
        #gridm = (1, 1)

        func = mod.get_function("nufft1")
        func(cuda.In(fourier_pts), cuda.In(ker), cuda.In(mu), np.int32(offset), np.int32(q), np.int32(M), np.int32(m),
             cuda.Out(tau),
             block=bdim, grid=gridm)

        return tau
        return res, 0

    @staticmethod
    def kernel_nufft(alpha, omega, M, precision='single'):
        if precision == 'single':
            b = 0.5993
            m=2
            q=10

        elif precision == 'double':
            b = 1.5629
            m=2
            q=28

        else:
            raise "precision is not legall!"
        n = len(alpha)
        mu = np.round(omega * m)

        P = calculate_kernel(omega, b, m, q)

        tau = np.zeros(m*M)
        offset1 = np.ceil((m*M-1)/2.)

        # calculating tau
        for k in range(n):
            for j in range(-q/2, q/2 + 1):
                idx = mu[k] + j
                idx = ((idx + offset1) % (M*m))
                tau[int(idx)] = tau[int(idx)] + P[k, j+q/2] * alpha[k]

        T = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(tau)))
        T = T * len(T)

        low_idx_M = int(-np.ceil((M-1)/2.))
        high_idx_M = int(np.floor((M-1)/2.)) + 1
        idx = np.arange(low_idx_M, high_idx_M)
        E = np.exp(b*(2*np.pi*idx/(m*M))**2)
        E = E.flatten(order='F')
        offset2 = offset1 + low_idx_M
        f = T[int(offset2):int(offset2 + M)] * E

        return f


if __name__ == "__main__":
    n = 10
    alpha = np.arange(-n/2, n/2) / float(n)
    omega = np.arange(-n/2, n/2)
    ret = nufft_gpu.kernel_nufft(alpha, omega, n)

    tau = nufft_gpu.forward1d(omega, alpha)

    print ret

