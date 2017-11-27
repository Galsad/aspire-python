import pycuda.driver as cuda
#import must stay here even if it's not used directly!
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.gpuarray as gpuarray
from lib import nudft
import nufft_cims

MAX_THREADS_PER_BLOCK = 1024
MAX_BLOCK_DIM = 1024

class nudft_gpu():
    @staticmethod
    def forward1d(sig, fourier_pts, eps=None):
        # converting all variables to complex one
        sig = sig.astype(np.complex64)
        fourier_pts = fourier_pts.astype(np.complex64)
        # grid = np.arange(np.ceil(-len(sig) / 2.), np.ceil(len(sig) / 2.)).astype(
        #     np.complex64)

        sz = np.uint32(sig.size)
        res = np.zeros_like(sig).astype(np.complex64)
        grid = np.zeros_like(sig).astype(np.complex64)

        # the kernel
        mod = SourceModule("""
        #include <pycuda-complex.hpp>
        #include <stdio.h>
        __global__ void dft1(pycuda::complex<float> *signal, pycuda::complex<float> *fourier_pts, pycuda::complex<float> *grid,int sz, pycuda::complex<float> *res)
        {
            // initialize grid
            int i = 0;
            for (i=0; i<sz; i++){
                grid[i] = i - (sz/2);
            }
            
            int idx = (threadIdx.x + blockDim.x * blockIdx.x) + 
            (threadIdx.y + blockDim.y * blockIdx.y) + 
            (threadIdx.z + blockDim.z * blockIdx.z);
            
            pycuda::complex<float> j(0, -1);
            pycuda::complex<float> tmp(0, 0);
    
            for (int i=0; i<sz; i++){
                tmp += exp(j * grid[i] * fourier_pts[idx]) * signal[i];
            }
            res[idx] = tmp;
        }
        """)

        bdim = (32, 32, 1)

        gridm = (sz / bdim[0] + (sz % bdim[0] > 0), 1)

        func = mod.get_function("dft1")
        func(cuda.In(sig), cuda.In(fourier_pts), cuda.In(grid), sz, cuda.Out(res),
             block=bdim, grid=gridm)
        return res

    @staticmethod
    def adjoint1d(sig_f, fourier_pts, eps=None):
        # converting all variables to complex one
        sig_f = sig_f.astype(np.complex64)
        fourier_pts = fourier_pts.astype(np.complex64)
        sz = np.uint32(sig_f.size)
        res = np.zeros_like(sig_f).astype(np.complex64)
        grid = np.zeros_like(sig_f).astype(np.complex64)

        # the kernel
        mod = SourceModule("""
            #include <pycuda-complex.hpp>
            #include <stdio.h>
            __global__ void adft1(pycuda::complex<float> *sig_f, pycuda::complex<float> *fourier_pts, pycuda::complex<float> *grid,int sz, pycuda::complex<float> *res)
            {
                
                // initialize grid
                int i = 0;
                for (i=0; i<sz; i++){
                    grid[i] = i - (sz/2);
                }
                
                int idx = (threadIdx.x + blockDim.x * blockIdx.x) + 
                (threadIdx.y + blockDim.y * blockIdx.y) + 
                (threadIdx.z + blockDim.z * blockIdx.z); //
    
                pycuda::complex<float> j(0, 1);
                pycuda::complex<float> tmp(0, 0);
    
                for (int i=0; i<sz; i++){
                    tmp += exp(j * grid[idx] * fourier_pts[i]) * sig_f[i];
                }
                res[idx] = tmp;
            }
            """)
        bdim = (32, 32, 1)

        gridm = (sz / bdim[0] + (sz % bdim[0] > 0), 1)

        func = mod.get_function("adft1")
        func(cuda.In(sig_f), cuda.In(fourier_pts), cuda.In(grid), sz, cuda.Out(res),
             block=bdim, grid=gridm)
        return res

    @staticmethod
    def forward2d(im, fourier_pts, eps=None):
        # converting all variables to complex one
        im = im.astype(np.complex64)
        fourier_pts = fourier_pts.astype(np.complex64)
        # grid = np.arange(np.ceil(-len(sig) / 2.), np.ceil(len(sig) / 2.)).astype(
        #     np.complex64)

        sz = np.uint32(im.shape[0])

        res = np.zeros_like(im[0]).astype(np.complex64)
        #grid = np.zeros_like(im).astype(np.complex64)

        grid = np.arange(np.ceil(-int(sz) / 2.), np.ceil(int(sz) / 2.)).astype(np.complex64)
        # grid_y and grid_x are like matlab conventions
        grid_x, grid_y = np.meshgrid(grid, grid)

        grid_x = grid_x.flatten().astype(np.complex64)
        grid_y = grid_y.flatten().astype(np.complex64)

        fourier_pts_x = fourier_pts[:, 0].astype(np.complex64)
        fourier_pts_y = fourier_pts[:, 1].astype(np.complex64)


        # pts = np.array([grid_x.flatten(), grid_y.flatten()]).astype(
        #     np.complex64)



        # the kernel
        mod = SourceModule("""
            #include <pycuda-complex.hpp>
            #include <stdio.h>
            __global__ void dft2(pycuda::complex<float> *im, pycuda::complex<float> *fourier_pts_x, pycuda::complex<float> *fourier_pts_y, pycuda::complex<float> *grid_x, pycuda::complex<float> *grid_y,int sz, pycuda::complex<float> *res)
            {
                // initialize grid
                //int i = 0;
                //for (i=0; i<sz; i++){
                //    grid[i] = i - (sz/2);
                //}

                int idx = (threadIdx.x + blockDim.x * blockIdx.x) + 
                (threadIdx.y + blockDim.y * blockIdx.y) + 
                (threadIdx.z + blockDim.z * blockIdx.z);

                pycuda::complex<float> j(0, -1);
                pycuda::complex<float> tmp(0, 0);

                for (int i=0; i < (sz*sz) ; i++){
                    tmp += exp(j * (grid_x[i] * fourier_pts_x[idx] + grid_y[i] * fourier_pts_y[idx])) * im[i];
                }
                res[idx] = tmp;
            }
            """)

        bdim = (32, 32, 1)

        gridm = (sz / bdim[0] + (sz % bdim[0] > 0), 1)

        func = mod.get_function("dft2")
        func(cuda.In(im.flatten()), cuda.In(fourier_pts_x),
             cuda.In(fourier_pts_y), cuda.In(grid_x), cuda.In(grid_y), sz,
             cuda.Out(res),
             block=bdim, grid=gridm)
        return res


if __name__ == "__main__":
    n = 8

    fourier_pts_x = np.random.uniform(-np.pi, np.pi, n)
    fourier_pts_y = np.random.uniform(-np.pi, np.pi, n)

    fourier_pts = np.array(zip(fourier_pts_x, fourier_pts_y))
    im = np.random.uniform(-1, 1, n * n).reshape(n, n)

    python_results = nudft.nudft2(im, fourier_pts)

    #py_nufft_obj = py_nufft.factory("gpu_dft")

    res = nudft_gpu.forward2d(im, fourier_pts)

    diff = 1e-08
    print nudft.nudft2(im, fourier_pts)
    print res
    print np.abs(np.sum(np.square(python_results - res))) / n