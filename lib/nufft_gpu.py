import pycuda.driver as cuda
#import must stay here even if it's not used directly!
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.gpuarray as gpuarray
from lib import nudft
import nufft_cims
from pycuda.elementwise import ElementwiseKernel


def calculate_kernel(omega, b, m, q):
    n = len(alpha)
    mu = np.round(omega * m)
    P = np.zeros([n, q + 1]).astype(np.complex64)

    # creating kernel function
    for i in range(-q / 2, q / 2 + 1):
        P[:, i + q / 2] = (np.exp(
            -(m * omega - (mu + i)) ** 2 / (4 * b))) / (
                              2 * np.sqrt(b * np.pi))

    return P

class nufft_gpu():


    # TODO - fix problem with dimension of kernel function
    @staticmethod
    def forward1d(fourier_pts, sig, eps=None):

        # preparing kernel #
        b = 0.5993
        m = 2
        q = 10

        sig = sig.astype(np.complex64)

        # kernel is of size q * (q+1)
        ker = calculate_kernel(sig, b, m, q)

        n = len(fourier_pts)

        # TODO - verify it
        M = n
        mu = np.ascontiguousarray(np.round(sig * m).astype(np.int32))
        tau_im = np.ascontiguousarray(np.zeros(M * m).astype(np.float32))
        tau_re = np.ascontiguousarray(np.zeros(M * m).astype(np.float32))
        # print tau
        offset = np.ceil((m * M - 1) / 2.)
        # ker = ker.astype(np.complex64)

        # converting all variables to complex one
        fourier_pts = np.ascontiguousarray(fourier_pts.astype(np.complex64))

        # the GPU kernel
        mod = SourceModule("""
        #include <pycuda-complex.hpp>
        #include <stdio.h>
        __global__ void nufft1(float *fourier_pts_re, float *fourier_pts_im, float *ker_re, float *ker_im, int *mu, int offset, int q, int M, int m, float *tau_re, float *tau_im)
        {
            int idx = (threadIdx.x + blockDim.x * blockIdx.x) + 
            (threadIdx.y + blockDim.y * blockIdx.y) + 
            (threadIdx.z + blockDim.z * blockIdx.z);
            
            idx = (threadIdx.x + blockDim.x * blockIdx.x) + (threadIdx.y + blockDim.y * blockIdx.y)*gridDim.x; 
            
            int i = -q / 2;
            int inner_idx = 0;
            
            float tmp_re = 0;
            float tmp_im = 0;
            
            
            for (i = -q/2; i < q/2 + 1 ;i++){
                inner_idx = mu[idx] + i;
                inner_idx = (inner_idx + offset + (m * M)) % (M * m);
                
                tmp_re = (ker_re[(idx*(q + 1)) + (i + q/2)] * fourier_pts_re[idx]) - (ker_im[(idx*(q + 1)) + (i + q/2)] * fourier_pts_im[idx]);
                tmp_im = (ker_re[(idx*(q + 1)) + (i + q/2)] * fourier_pts_im[idx]) + (ker_im[(idx*(q + 1)) + (i + q/2)] * fourier_pts_re[idx]);
                
                atomicAdd(&tau_im[inner_idx], tmp_im);
                atomicAdd(&tau_re[inner_idx], tmp_re);
                
            }
        }
        """)

        bdim = (n, 1, 1)

        gridm = (n / bdim[0] + (n % bdim[0] > 0), 1)

        #print gridm
        #gridm = (1, 1)

        func = mod.get_function("nufft1")

        #print ker

        ker = ker.flatten()

        ker_re = np.ascontiguousarray(ker.real).astype(np.float32)
        ker_im = np.ascontiguousarray(ker.imag).astype(np.float32)


        fourier_pts_re = np.ascontiguousarray(fourier_pts.real).astype(np.float32)
        fourier_pts_im = np.ascontiguousarray(fourier_pts.imag).astype(np.float32)

        #print "ker is:"
        #print ker_re
        #print fourier_pts_re[:50]

        func(cuda.In(fourier_pts_re), cuda.In(fourier_pts_im), cuda.In(ker_re), cuda.In(ker_im), cuda.In(mu), np.int32(offset), np.int32(q), np.int32(M), np.int32(m),
             cuda.Out(tau_re), cuda.Out(tau_im),
             block=bdim, grid=gridm)

        print tau_re


        tau = tau_re + 1j * tau_im

        T = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(tau)))
        T = T * len(T)

        low_idx_M = int(-np.ceil((M - 1) / 2.))
        high_idx_M = int(np.floor((M - 1) / 2.)) + 1
        idx = np.arange(low_idx_M, high_idx_M)
        E = np.exp(b * (2 * np.pi * idx / (m * M)) ** 2)
        E = E.flatten(order='F')
        offset2 = offset + low_idx_M
        f = T[int(offset2):int(offset2 + M)] * E

        return f


        # return zip(tau_re, tau_im)
        #return res, 0

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

        tau = np.zeros(m*M).astype(np.complex64)
        offset1 = np.ceil((m*M-1)/2.)

        # calculating tau
        for k in range(n):
            for j in range(-q/2, q/2 + 1):
                idx = mu[k] + j
                idx = ((idx + offset1) % (M*m))
                # print idx
                tau[int(idx)] = tau[int(idx)] + P[k, j+q/2] * alpha[k]

        print "***"
        print tau
        print "***"

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

    ret = nufft_gpu.kernel_nufft(alpha.astype(np.complex64), omega, n)


    tau = nufft_gpu.forward1d(alpha, omega)

    #print tau
    #print ret

