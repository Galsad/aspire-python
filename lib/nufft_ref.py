import numpy as np
import nufft_cims
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import time

b = 0.5993
m = 2
BLOCK_SIZE = 1024

def calculate_kernel(omega, b, m, q, n):
    mu = np.round(omega * m)
    P = np.zeros([n, q + 1]).astype(np.complex64)

    # creating kernel function
    for i in range(-q / 2, q / 2 + 1):
        P[:, i + q / 2] = (np.exp(
            -(m * omega - (mu + i)) ** 2 / (4 * b))) / (
                              2 * np.sqrt(b * np.pi))

    return P


def kernel_nufft_1d(alpha, omega, M, precision='single'):
    '''
    reference code for GPU nufft
    :param alpha:
    :param omega:
    :param M:
    :param precision:
    :return:
    '''
    if precision == 'single':
        b = 0.5993
        m = 2
        q = 10

    elif precision == 'double':
        b = 1.5629
        m = 2
        q = 28

    else:
        raise "precision is not legall!"
    n = len(alpha)
    mu = np.round(omega * m)

    # P = calculate_kernel(omega, b, m, q, n)

    tau = np.zeros(m * M).astype(np.complex64)
    offset1 = np.ceil((m * M - 1) / 2.)

    # calculating tau
    for k in range(n):
        for j in range(-q / 2, q / 2 + 1):
            idx = mu[k] + j
            idx = ((idx + offset1) % (M * m))
            # tau[int(idx)] = tau[int(idx)] + P[k, j+q/2] * alpha[k]
            tau[int(idx)] = tau[int(idx)] + ((np.exp( -(m * omega[k] - (mu[k] + j)) ** 2 / (4 * b))) / ( 2 * np.sqrt(b * np.pi))) * alpha[k]

    T = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(tau)))
    T = T * len(T)

    low_idx_M = int(-np.ceil((M - 1) / 2.))
    high_idx_M = int(np.floor((M - 1) / 2.)) + 1
    idx = np.arange(low_idx_M, high_idx_M)
    E = np.exp(b * (2 * np.pi * idx / (m * M)) ** 2)
    E = E.flatten(order='F')
    offset2 = offset1 + low_idx_M
    f = T[int(offset2):int(offset2 + M)] * E

    return f


def kernel_nufft_2d(alpha, omega, M, precision='single'):
    '''
     reference code for GPU nufft
     :param alpha:
     :param omega: 2 x n array of sampling points
     :param M:
     :param precision:
     :return:
     '''
    if precision == 'single':
        b = 0.5993
        m = 2
        q = 10

    elif precision == 'double':
        b = 1.5629
        m = 2
        q = 28

    n = len(alpha)
    mu = np.round(omega * m)

    tau = np.zeros([m * M, m * M]).astype(np.complex64)
    offset1 = np.ceil((m * M - 1) / 2.)

    for k in range(n):
        for j1 in range(-q/2, q/2 + 1):
            for j2 in range(-q/2, q/2 + 1):
                idx = mu[k] + (j1, j2)
                idx = (idx + offset1) % (M*m)

                tmp = -((m*omega[k, 0] - (mu[k, 0] + j1))**2 + (m*omega[k, 1] - (mu[k, 1] + j2))**2) / (4*b)
                tmp2 = (np.exp(tmp) / (4 * b * np.pi)) * alpha[k]

                tau[int(idx[0]), int(idx[1])] = tau[int(idx[0]), int(idx[1])] + tmp2

    T = np.fft.ifftshift(tau)
    T = np.fft.ifft2(T)
    T = np.fft.ifftshift(T)

    T = T * len(T) *len(T)

    low_idx_M = -np.ceil((M - 1) / 2.)
    high_idx_M = np.floor((M - 1) / 2.) + 1
    idx = np.arange(low_idx_M, high_idx_M)
    E = np.exp(b * (2. * np.pi * idx / (m * M)) ** 2)
    E = E.flatten(order='F')
    E = np.outer(E ,E)

    offset2 = offset1 + low_idx_M

    f = T[int(offset2):int(offset2) + M , int(offset2) : int(offset2) + M] * E
    return f, 0

def slow_forward1d(alpha, omega, eps=None):
    '''
    running 1dfft on GPU:
    calculating the sum:

            n
    f(j) = sum alpha(k)*exp(2*pi*i*j*omega(k)/M)
           k=1

    :param alpha: Coefficients in the sums above. Real or complex numbers.
    :param omega: Sampling frequnecies. Real numbers in the range [-n/2,n/2]
    :param eps:
    :return: the sum defined above
    '''

    # kernel parameters (single precision)#

    omega = omega.astype(np.float32)
    n = len(alpha)

    M = n
    mu = np.ascontiguousarray(np.round(omega * m).astype(np.int32))
    offset = np.ceil((m * M - 1) / 2.)

    tau_im = pycuda.gpuarray.zeros(M * m, dtype=np.float32)
    tau_re = pycuda.gpuarray.zeros(M * m, dtype=np.float32)

    alpha = np.ascontiguousarray(alpha.astype(np.complex64))

    # the GPU kernel
    mod = SourceModule("""
    #include <pycuda-complex.hpp>
    #include <stdio.h>
    __global__ void nufft1(pycuda::complex<float> *alpha, float *omega, int precision ,int *mu, int offset, int M, float* tau_re, float* tau_im)
    { 
        int k = (threadIdx.x + blockDim.x * blockIdx.x); 

        if (k >= M){
            return ;
        }

        // kernel variables
        int q = 0;
        int m = 0;
        float b = 0;

        // single precision
        if (precision == 0){
            b = 0.5993;
            m=2;
            q=10;
        }

        if (precision == 1){
            b = 1.5629;
            m=2;
            q=28;
        }

        int j = -q / 2;
        int idx = 0;

        // consts
        const float pi = 3.141592653589793;
        const pycuda::complex<float> comp_m(m, 0);
        const pycuda::complex<float> denominator((2 * powf(b * pi, 0.5)), 0);

        // inner loop variables
        pycuda::complex<float> tmp(0, 0);
        pycuda::complex<float> add;
        pycuda::complex<float> numerator;

        for (j = -q/2; j < q/2 + 1;j++){
            idx = mu[k] + j;
            idx = (idx + offset + (m * M)) % (M * m);

            tmp.real(j + mu[k]);
            numerator = ((comp_m * omega[k] - tmp) * (comp_m * omega[k] - tmp) / (4*b));
            add = (exp(-numerator) / denominator) * alpha[k];

            atomicAdd(&tau_im[idx], imag(add));
            atomicAdd(&tau_re[idx], real(add));                
        }
    }
    """)

    bdim = (BLOCK_SIZE, 1, 1)
    gridm = (n / bdim[0] + (n % bdim[0] > 0), 1, 1)

    func = mod.get_function("nufft1")

    func(cuda.In(alpha), cuda.In(omega), np.int32(0), cuda.In(mu),
         np.int32(offset), np.int32(M), tau_re, tau_im,
         block=bdim, grid=gridm)

    tau = tau_re.get() + 1j * tau_im.get()
    T = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(tau)))
    T = T * len(T)

    low_idx_M = int(-np.ceil((M - 1) / 2.))
    high_idx_M = int(np.floor((M - 1) / 2.)) + 1
    idx = np.arange(low_idx_M, high_idx_M)
    E = np.exp(b * (2 * np.pi * idx / (m * M)) ** 2)
    E = E.flatten(order='F')
    offset2 = offset + low_idx_M
    f = T[int(offset2):int(offset2 + M)] * E

    return f, 0


if __name__ == "__main__":

    ### testing 1D ###
    # n = 20

    # alpha = np.arange(-n / 2, n / 2) / float(n)
    # alpha = np.random.uniform(-np.pi, np.pi, n)
    # omega = np.arange(-n / 2, n / 2)

    # ret = kernel_nufft_1d(alpha, omega, n)

    ### testing 2D ###
    n = 128

    alpha = np.arange(-n/2, n/2) / float(n)
    #alpha = np.random.uniform(-np.pi, np.pi, n)
    omega_x = np.arange(-n/2, n/2)
    omega_y = np.arange(-n/2, n/2)

    omega = np.array([omega_x, omega_y]).transpose()


    ret = kernel_nufft_2d(alpha, omega, n)

    # print ret
