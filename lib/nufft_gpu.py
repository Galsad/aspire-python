import numba
import numpy as np
# import must stay here even if it's not used directly!
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import nufft_cims
import nufft_ref
import time
import pyfftw
import multiprocessing

import skcuda
import skcuda.linalg
import skcuda.fft as cu_fft

# import pytest_benchmark

BLOCK_SIZE = 1024
start = cuda.Event()
end = cuda.Event()
num_threads = multiprocessing.cpu_count()

# GLOBAL CONSTS
b = 0.5993
m = 2

# the GPU kernel
mod = SourceModule("""
#include <pycuda-complex.hpp>
#include <stdio.h>
__global__ void fast_nufft1(float *alpha_re, float *alpha_im, float *omega, int M, float* tau_re, float* tau_im)
{ 
    int k; k = (threadIdx.x + blockDim.x * blockIdx.x); 
    int j; j = (threadIdx.y + blockDim.y * blockIdx.y);

    if (k >= M){
        return ;
    }

    // always single precision 
    float b = 0.5993;
    int m=2;
    int q=10;   
    j -= q/2;
    // consts
    const float pi = 3.141592653589793;
    const float denominator = 2 * powf  (b * pi, 0.5);
    float numerator = 0;
    float add_Re = 0;
    float add_Im = 0;
    //int j = 0;
    int idx = 0;

    // m is always even - so this is the right formula
    int offset = (M * m - 1)/2 + 1;
    //for (j = -q/2; j < q/2 + 1;j++){
        idx = lround(omega[k] * 2) + j;

        numerator = exp(-1*(((m * omega[k] - idx) * (m * omega[k] - idx) / (4*b)))) / denominator;
        idx = (int) fmod((float)(idx + offset + (m * M)), (float) (M * m));

        add_Re = numerator * alpha_re[k];
        add_Im = numerator * alpha_im[k];
        //tau_im[idx] += add_Im;
        //tau_re[idx] += add_Re;
        atomicAdd(&tau_im[idx], add_Im);
        atomicAdd(&tau_re[idx], add_Re);
    //}                       
}
""")

gpu_fft1 = mod.get_function("fast_nufft1")

mod2 = SourceModule("""
        #include <pycuda-complex.hpp>
        #include <stdio.h>
        __global__ void nufft2(int M, int offset, pycuda::complex<float> *alpha, float *omega_x, float *omega_y, int* mu_x, int* mu_y, float* tau_re, float* tau_im)
        { 
            int k = (threadIdx.x + blockDim.x * blockIdx.x); 
            if (k >= M){
                return ;
            }
            float b = 0.5993;
            int m=2;
            int q=10;
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
                    //tau_im[idx1 * (M * m) + idx2] = tau_im[idx1 * (M * m) + idx2] + imag(add);
                    //tau_re[idx1 * (M * m) + idx2] = tau_re[idx1 * (M * m) + idx2] + real(add);
                    atomicAdd(&tau_im[idx1 * (M * m) + idx2], imag(add));
                    atomicAdd(&tau_re[idx1 * (M * m) + idx2], real(add));  
                }              
            }
        }
        """)

gpu_fft2 = mod2.get_function("nufft2")

mod2 = SourceModule("""
        #include <pycuda-complex.hpp>
        #include <stdio.h>
        
        __global__ void fast_nufft2(int M, float *alpha_re, float *alpha_im, float *omega_x, float *omega_y,float *tau_re, float *tau_im)
        { 
            int k; k = (threadIdx.x + blockDim.x * blockIdx.x);
            int j1; j1 = (threadIdx.y + blockDim.y * blockIdx.y);
            int j2; j2 = (threadIdx.z + blockDim.z * blockIdx.z); 
            float b = 0.5993;
            int m=2;
            int q=10;
            
            if (k >= M){
                // printf("Index is out of bound\\n");
                return ;
            }
            

            j1 = j1 - q/2;
            j2 = j2 - q/2;
            
            int idx1 = 0;
            int idx2 = 0;
            
            // consts
            const float pi = 3.141592653589793;
            const float denominator = 4 * b * pi;
            float add_Re = 0;
            float add_Im = 0;
            
            pycuda::complex<float> add;
            
            int offset = (M*m - 1)/2 + 1;
            float numerator = 0;

            float omega_x_k = omega_x[k];
            float omega_y_k = omega_y[k];            


            idx1 = lround(omega_x_k * m) + j1;
            idx2 = lround(omega_y_k * m) + j2;
                    
            numerator = exp(-1*(((m * omega_x_k - idx1) * (m * omega_x_k - idx1) + (m * omega_y_k - idx2) * (m * omega_y_k - idx2)) / (4*b))) / denominator;
                     
            idx1 = (int)fmod((float)(idx1 + offset + (m * M)), (float)(M*m));
            idx2 = (int)fmod((float)(idx2 + offset + (m * M)), (float)(M*m));

            add_Re = numerator * alpha_re[k];
            add_Im = numerator * alpha_im[k];

            atomicAdd(&tau_im[idx1 * (M * m) + idx2], add_Im);
            atomicAdd(&tau_re[idx1 * (M * m) + idx2], add_Re);
        }
        """)
fast_gpu_fft2 = mod2.get_function("fast_nufft2")

mod4 = SourceModule("""
    #include <stdio.h>
    #include <pycuda-complex.hpp>
    __global__ void fftshift(pycuda::complex<float>* Source, pycuda::complex<float>* Destination, int size)
    {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) % size;
        int j = (blockIdx.x * blockDim.x + threadIdx.x) / size;
        
        // new indices
        int xc = i + size/2;
        int yc = j + size/2;
        
        // index is bounded by size
        if (xc >= size){
            xc -= size;
        }

        if (yc >= size){
            yc -= size;
        }

        Destination[xc + yc*size].real(real(Source[i + j*size]));
        Destination[xc + yc*size].imag(imag(Source[i + j*size]));
    }
    """)

my_fftshift = mod4.get_function("fftshift")

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
        omega = omega.astype(np.float32)
        n = len(alpha)

        M = n
        offset = np.ceil((m * M - 1) / 2.)

        tau_im = pycuda.gpuarray.empty(M * m, dtype=np.float32)
        tau_re = pycuda.gpuarray.empty(M * m, dtype=np.float32)

        bdim = (BLOCK_SIZE / 10, 10, 1)
        gridm = ((n / BLOCK_SIZE + (n % (BLOCK_SIZE) > 0)), 1, 1)

        alpha_real = np.ascontiguousarray(alpha.real)
        alpha_imag = np.ascontiguousarray(alpha.imag)

        gpu_fft1(cuda.In(alpha_real), cuda.In(alpha_imag), cuda.In(omega),
                 np.int32(M), tau_re, tau_im,
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
        t0 = time.time()

        # prepare parameters for running
        omega2 = omega.astype(np.float32)
        alpha_real2 = np.ascontiguousarray(alpha.real)
        alpha_imag2 = np.ascontiguousarray(alpha.imag)
        M = len(alpha)
        global size
        size = M * m

        # allocating memory for tau if it's the first time this code runs
        if "tau_im_gpu" not in globals() and "tau_re_gpu" not in globals():
            global tau_im_gpu
            global tau_re_gpu
            tau_im_gpu = gpuarray.to_gpu(np.zeros([size * size], dtype=np.float32))
            tau_re_gpu = gpuarray.to_gpu(np.zeros([size * size], dtype=np.float32))
            print "some memory was allocated on GPU"

        else:
            tau_im_gpu.fill(0)
            tau_re_gpu.fill(0)

        # allocating memory for results if it's the first time this code runs
        if "T_res_gpu " not in globals() and "T_res_gpu2" not in globals():
            global T_res_gpu
            global T_res_gpu2
            T_res_gpu = gpuarray.to_gpu(np.ascontiguousarray(np.zeros([size, size], dtype=np.complex64)))
            T_res_gpu2 = gpuarray.to_gpu(np.ascontiguousarray(np.zeros([size ,size], dtype=np.complex64)))
            print "some memory was allocated on GPU"

        else:
            # fill arrays in 0's for reuse
            T_res_gpu.fill(0)
            T_res_gpu2.fill(0)

        # the GPU kernel
        bdim = (BLOCK_SIZE/121, 11, 11)
        gridm = ((M / bdim[0] + (M % bdim[0] > 0)), 1, 1)

        fast_gpu_fft2(np.int32(M), cuda.In(alpha_real2), cuda.In(alpha_imag2),
                      cuda.In(omega2[:, 0]), cuda.In(omega2[:, 1]), tau_re_gpu, tau_im_gpu,
                      block=bdim, grid=gridm)

        tau_gpu = tau_re_gpu + 1j*tau_im_gpu

        bdim = (min(1024, size*size), 1, 1)
        gridm = ((len(tau_re_gpu) / bdim[0] + (len(tau_re_gpu) % bdim[0] > 0)), 1, 1)

        my_fftshift(tau_gpu, T_res_gpu2, np.int32(size), block=bdim, grid=gridm)

        if "plan" not in globals():
            print "Planning"
            global plan
            plan = cu_fft.Plan((size, size), np.complex64, np.complex64)

        if plan.shape != (size, size):
            print "Planning"
            global plan
            plan = cu_fft.Plan((size, size), np.complex64, np.complex64)

        cu_fft.ifft(T_res_gpu2, T_res_gpu, plan, True)
        my_fftshift(T_res_gpu, T_res_gpu2, np.int32(size),
                    block=bdim, grid=gridm)

        T_res_gpu2 *= (size * size)

        bound = (M - 1) / 2.
        low_idx_M = -np.ceil(bound)
        high_idx_M = int(bound) + 1

        idx = np.arange(low_idx_M, high_idx_M)
        E = np.exp(b * (2. * np.pi * idx / (size)) ** 2)
        E = np.outer(E, E)

        offset = int(np.ceil((size - 1) / 2.) + low_idx_M)
        offset2 = offset + M

        T = T_res_gpu2.get()
        f = T[offset:offset2, offset: offset2] * E

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


# def test_my_stuff(benchmark, alpha, omega, inner_block_size=16):
#    result = benchmark(nufft_gpu.fast_forward1d,arg=(alpha, omega, inner_block_size), iterations=10, rounds=100)


if __name__ == "__main__":
    # delete global variables
    for x in globals():
        del x

    for i in range(6):
        n = 33

        #alpha = np.arange(-n / 2, n / 2) / float(n)
        alpha = np.random.uniform(-np.pi, np.pi, n)
        alpha = alpha.astype(np.complex64)
        #omega_x = np.arange(-n / 2, n / 2)
        omega_x = np.random.uniform(-n/2, n/2, n)
        #omega_y = np.arange(-n / 2, n / 2)
        omega_y = np.random.uniform(-n / 2, n / 2, n)
        omega = np.array([omega_x, omega_y]).transpose()

        #    test_my_stuff(benchmark, alpha, omega, 32)

        ret = nufft_gpu.forward2d(alpha, omega)
        ret2 = nufft_ref.kernel_nufft_2d(alpha, omega, n)

        print np.abs(np.sum(np.square(ret[0] - ret2[0]))) / (
            len(ret[0]) * len(ret[0]))

# ret = nufft_gpu.forward1d(alpha, omega)
#    ret2 = nufft_ref.slow_forward1d(alpha, omega)

#    print np.abs(np.sum(np.square(ret[0] - ret2[0]))) / (len(ret[0]))
