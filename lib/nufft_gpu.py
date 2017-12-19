import pycuda.driver as cuda
#import must stay here even if it's not used directly!
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.gpuarray as gpuarray
from lib import nudft
import nufft_cims

import nufft_ref

import pyfft



BLOCK_SIZE = 1024

class nufft_gpu():

    @staticmethod
    def forward1d(alpha, omega, eps=None):
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
        b = 0.5993
        m = 2

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

        func(cuda.In(alpha), cuda.In(omega), np.int32(0), cuda.In(mu), np.int32(offset), np.int32(M), tau_re, tau_im,
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

    @staticmethod
    def forward2d(alpha, omega, eps=None):
        b = 0.5993
        m = 2

        omega = omega.astype(np.float32)
        n = len(alpha)

        M = n
        mu = np.round(omega * m).astype(np.int32)

        mu_x = np.ascontiguousarray(mu[ :,0].astype(np.int32))
        mu_y = np.ascontiguousarray(mu[ :,1].astype(np.int32))

        offset = np.ceil((m * M - 1) / 2.)

        tau_im = pycuda.gpuarray.zeros([M * m * M * m], dtype=np.float32)
        tau_re = pycuda.gpuarray.zeros([M * m * M * m], dtype=np.float32)

        alpha = np.ascontiguousarray(alpha.astype(np.complex64))

        # the GPU kernel
        mod = SourceModule("""
                #include <pycuda-complex.hpp>
                #include <stdio.h>
                __global__ void nufft2(int M, int offset, pycuda::complex<float> *alpha, float *omega_x, float *omega_y, int* mu_x, int* mu_y, float* tau_re, float* tau_im)
                { 
                    int k = (threadIdx.x + blockDim.x * blockIdx.x); 
                    
                    if (k >= M){
                        return ;
                    }
                    
                    // kernel variables
                    //int q = 0;
                    //int m = 0;
                    //float b = 0;

                    // single precision
                    //if (precision == 0){
                        float b = 0.5993;
                        int m=2;
                        int q=10;
                    //}

                    //if (precision == 1){
                    //    b = 1.5629;
                    //    m=2;
                    //    q=28;
                    //}

                    int j1 = -q / 2;
                    int j2 = -q / 2;
                    int idx1 = 0;
                    int idx2 = 0;

                    // consts
                    const float pi = 3.141592653589793;
                    const pycuda::complex<float> comp_m(m, 0);
                    const pycuda::complex<float> denominator(4 * b * pi, 0);

                    // inner loop variables
                    pycuda::complex<float> tmp1(0, 0);
                    pycuda::complex<float> tmp2(0, 0);
                    
                    pycuda::complex<float> add;
                    pycuda::complex<float> numerator;
                    
                    for (j1 = -q/2; j1 < q/2 + 1; j1++){
                        for (j2 = -q/2; j2 < q/2 + 1; j2++ ){
                        
                            idx1 = mu_x[k] + j1;
                            idx2 = mu_y[k] + j2;
                            
                            idx1 = (idx1 + offset + (m * M)) % (M * m);
                            idx2 = (idx2 + offset + (m * M)) % (M * m);

                            tmp1.real(j1 + mu_x[k]);
                            tmp2.real(j2 + mu_y[k]);
                            
                            numerator = ((comp_m * omega_x[k] - tmp1) * (comp_m * omega_x[k] - tmp1) + (comp_m * omega_y[k] - tmp2) * (comp_m * omega_y[k] - tmp2)) / (4*b);
                            add = (exp(-numerator) / denominator) * alpha[k];
                            
                            tau_im[idx1 * (M * m) + idx2] = tau_im[idx1 * (M * m) + idx2] + imag(add);
                            tau_re[idx1 * (M * m) + idx2] = tau_re[idx1 * (M * m) + idx2] + real(add);
                            
                            
                            //atomicAdd(&tau_im[idx1 * (M * m) + idx2], imag(add));
                            //atomicAdd(&tau_re[idx1 * (M * m) + idx2], real(add));  
                        }              
                    }
                }
                """)

        bdim = (BLOCK_SIZE, 1, 1)
        gridm = (n / bdim[0] + (n % bdim[0] > 0), 1, 1)

        func = mod.get_function("nufft2")

        func(np.int32(M), np.int32(offset), cuda.In(alpha), cuda.In(omega[:, 0]),
             cuda.In(omega[:, 1]) , cuda.In(mu_x), cuda.In(mu_y), tau_re, tau_im,
             block=bdim, grid=gridm)

        tau = (tau_re.get() + 1j * tau_im.get())
        tau = tau.reshape(m*M, m*M, order='F')

        tau = tau.astype(np.complex64)

        T = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(tau)))

        T = T * len(T) * len(T)

        low_idx_M = -np.ceil((M - 1) / 2.)
        high_idx_M = np.floor((M - 1) / 2.) + 1
        idx = np.arange(low_idx_M, high_idx_M)
        E = np.exp(b * (2. * np.pi * idx / (m * M)) ** 2)
        E = E.flatten(order='F')
        E = np.outer(E, E)

        offset2 = offset + low_idx_M

        f = T[int(offset2):int(offset2) + M,
            int(offset2): int(offset2) + M] * E
        return f, 0

    @staticmethod
    def forward3d(fourier_pts, sig, eps=None):
        return 0, 0

    @staticmethod
    def adjoint1d(fourier_pts, sig, eps=None):
        return 0, 0

    @staticmethod
    def adjoint2d(fourier_pts, sig, eps=None):
        return 0, 0

    @staticmethod
    def adjoint3d(fourier_pts, sig, eps=None):
        return 0, 0


if __name__ == "__main__":
    n = 1024

    alpha = np.arange(-n / 2, n / 2) / float(n)
    # alpha = np.random.uniform(-np.pi, np.pi, n)
    omega_x = np.arange(-n / 2, n / 2)
    omega_y = np.arange(-n / 2, n / 2)
    omega = np.array([omega_x, omega_y]).transpose()


    ret = nufft_gpu.forward2d(alpha, omega)
    ret2 = nufft_ref.kernel_nufft_2d(alpha, omega, n)

    print np.abs(np.sum(np.square(ret[0] - ret2))) / (len(ret2)*len(ret2))




