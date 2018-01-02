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

#TODO -- anudft1, anudft2 works only if size == len(im/sig) -- need to fix it
#

class nudft_gpu():
    @staticmethod
    def forward1d(sig, fourier_pts , eps=None):
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
        return res, 0

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
        return res, 0

    @staticmethod
    def forward2d(im, fourier_pts, eps=None):
        # converting all variables to complex one
        im = im.astype(np.complex64)
        fourier_pts = fourier_pts.astype(np.complex64)
        # grid = np.arange(np.ceil(-len(sig) / 2.), np.ceil(len(sig) / 2.)).astype(
        #     np.complex64)

        sz = np.uint32(fourier_pts.shape[1])

        res = np.zeros_like(im[0]).astype(np.complex64)
        #grid = np.zeros_like(im).astype(np.complex64)

        grid = np.arange(np.ceil(-int(sz) / 2.), np.ceil(int(sz) / 2.)).astype(np.complex64)
        # grid_y and grid_x are like matlab conventions
        grid_x, grid_y = np.meshgrid(grid, grid)

        grid_x = grid_x.flatten(order='F').astype(np.complex64)
        grid_y = grid_y.flatten(order='F').astype(np.complex64)

        fourier_pts_x = fourier_pts[0].astype(np.complex64)
        fourier_pts_y = fourier_pts[1].astype(np.complex64)

        # grid_x = np.zeros(sz * sz).astype(np.complex64)
        # grid_y = np.zeros(sz * sz).astype(np.complex64)

        # pts = np.array([grid_x.flatten(), grid_y.flatten()]).astype(
        #     np.complex64)



        # the kernel
        mod = SourceModule("""
            #include <pycuda-complex.hpp>
            #include <stdio.h>
            __global__ void dft2(pycuda::complex<float> *im, pycuda::complex<float> *fourier_pts_x, pycuda::complex<float> *fourier_pts_y, pycuda::complex<float> *grid_x, pycuda::complex<float> *grid_y,int sz, pycuda::complex<float> *res)
            {
                // initialize grid
                //int i=0;
                //int k=0;
                //for (i=0; i<sz; i++){
                //   for (k=0; k<sz; k++){
                //        grid_x[i*sz + k] = i - (sz/2);
                //        grid_y[i*sz + k] = k - (sz/2);
                //        printf(" %d, ",  k - (sz/2));
                //    }
                //    printf("\\n");
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
        func(cuda.In(im.flatten(order='F')), cuda.In(fourier_pts_x),
             cuda.In(fourier_pts_y), cuda.In(grid_x), cuda.In(grid_y), sz,
             cuda.Out(res),
             block=bdim, grid=gridm)
        return res, 0

    @staticmethod
    def adjoint2d(im_f, fourier_pts , eps=None):
        # converting all variables to complex one
        im_f = im_f.astype(np.complex64)
        sz = len(im_f)
        fourier_pts = fourier_pts.astype(np.complex64)

        grid = np.arange(np.ceil(-sz / 2.), np.ceil(sz / 2.)).astype(
             np.complex64)
        grid_x, grid_y = np.meshgrid(grid, grid)

        res = np.zeros(sz * sz).astype(np.complex64)

        grid_x = grid_x.flatten().astype(np.complex64)
        grid_y = grid_y.flatten().astype(np.complex64)

        fourier_pts_x = fourier_pts[:, 0].astype(np.complex64)
        fourier_pts_y = fourier_pts[:, 1].astype(np.complex64)

        # the kernel
        mod = SourceModule("""
            #include <pycuda-complex.hpp>
            #include <stdio.h>
            __global__ void adft2(pycuda::complex<float> *im_f, pycuda::complex<float> *fourier_pts_x, pycuda::complex<float> *fourier_pts_y, pycuda::complex<float> *grid_x, pycuda::complex<float> *grid_y,int sz, pycuda::complex<float> *res)
            {
                // initialize grid
                //int i = 0;
                //for (i=0; i<sz; i++){
                //    grid[i] = i - (sz/2);
                //}

                int idx = (threadIdx.x + blockDim.x * blockIdx.x) + 
                (threadIdx.y + blockDim.y * blockIdx.y) + 
                (threadIdx.z + blockDim.z * blockIdx.z);
                
                // printf("index is %d\\n", idx); 

                pycuda::complex<float> j(0, 1);
                pycuda::complex<float> tmp(0, 0);

                for (int i=0; i < (sz) ; i++){
                    tmp += exp(j * (grid_x[idx] * fourier_pts_x[i] + grid_y[idx] * fourier_pts_y[i])) * im_f[i];
                }
                res[idx] = tmp;
            }
            """)

        bdim = (32, 32, 1)

        gridm = ((sz*sz) / bdim[0] + ((sz*sz) % bdim[0] > 0), 1)

        sz = np.uint32(sz)

        func = mod.get_function("adft2")
        func(cuda.In(im_f), cuda.In(fourier_pts_x),
             cuda.In(fourier_pts_y), cuda.In(grid_x), cuda.In(grid_y), sz,
             cuda.Out(res),
             block=bdim, grid=gridm)
        return res.reshape(sz, sz), 0


    @staticmethod
    def forward3d(vol, fourier_pts, eps=None):
        # converting all variables to complex one
        vol = vol.astype(np.complex64)
        fourier_pts = fourier_pts.astype(np.complex64)

        sz = np.uint32(fourier_pts.shape[1])

        res = np.zeros_like(vol[0][0]).astype(np.complex64)

        grid = np.arange(np.ceil(-int(sz) / 2.), np.ceil(int(sz) / 2.)).astype(np.complex64)
        # grid_y and grid_x are like matlab conventions
        grid_x, grid_y, grid_z = np.meshgrid(grid, grid, grid)

        grid_x = grid_x.flatten(order='F').astype(np.complex64)
        grid_y = grid_y.flatten(order='F').astype(np.complex64)
        grid_z = grid_z.flatten(order='F').astype(np.complex64)

        fourier_pts_x = fourier_pts[0].astype(np.complex64)
        fourier_pts_y = fourier_pts[1].astype(np.complex64)
        fourier_pts_z = fourier_pts[2].astype(np.complex64)

        # the kernel
        mod = SourceModule("""
            #include <pycuda-complex.hpp>
            #include <stdio.h>
            __global__ void dft3(pycuda::complex<float> *vol, pycuda::complex<float> *fourier_pts_x, pycuda::complex<float> *fourier_pts_y, pycuda::complex<float> *fourier_pts_z, pycuda::complex<float> *grid_x, pycuda::complex<float> *grid_y, pycuda::complex<float> *grid_z, int sz, pycuda::complex<float> *res)
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

                for (int i=0; i < (sz*sz*sz) ; i++){
                    tmp += exp(j * (grid_x[i] * fourier_pts_x[idx] + grid_y[i] * fourier_pts_y[idx] + grid_z[i] * fourier_pts_z[idx])) * vol[i];
                }
                res[idx] = tmp;
            }
            """)

        bdim = (32, 32, 1)

        gridm = (sz / bdim[0] + (sz % bdim[0] > 0), 1)

        func = mod.get_function("dft3")
        func(cuda.In(vol.flatten(order='F')), cuda.In(fourier_pts_y),
             cuda.In(fourier_pts_x), cuda.In(fourier_pts_z), cuda.In(grid_x),
             cuda.In(grid_y), cuda.In(grid_z), sz, cuda.Out(res),
             block=bdim, grid=gridm)
        return res, 0

    @staticmethod
    def adjoint3d(vol_f,fourier_pts , eps=None):
        # converting all variables to complex one
        sz = len(vol_f)
        vol_f = vol_f.astype(np.complex64)
        fourier_pts = fourier_pts.astype(np.complex64)

        grid = np.arange(np.ceil(-sz / 2.), np.ceil(sz / 2.)).astype(
            np.complex64)
        grid_x, grid_y, grid_z = np.meshgrid(grid, grid, grid)

        res = np.zeros(sz * sz * sz).astype(np.complex64)

        grid_x = grid_x.flatten().astype(np.complex64)
        grid_y = grid_y.flatten().astype(np.complex64)
        grid_z = grid_z.flatten().astype(np.complex64)

        fourier_pts_x = fourier_pts[:, 0].astype(np.complex64)
        fourier_pts_y = fourier_pts[:, 1].astype(np.complex64)
        fourier_pts_z = fourier_pts[:, 2].astype(np.complex64)

        # the kernel
        mod = SourceModule("""
            #include <pycuda-complex.hpp>
            #include <stdio.h>
            __global__ void adft3(pycuda::complex<float> *vol_f, pycuda::complex<float> *fourier_pts_x, pycuda::complex<float> *fourier_pts_y, pycuda::complex<float> *fourier_pts_z, pycuda::complex<float> *grid_x, pycuda::complex<float> *grid_y, pycuda::complex<float> *grid_z, int sz, pycuda::complex<float> *res)
            {
                // initialize grid
                //int i = 0;
                //for (i=0; i<sz; i++){
                //    grid[i] = i - (sz/2);
                //}

                int idx = (threadIdx.x + blockDim.x * blockIdx.x) + 
                (threadIdx.y + blockDim.y * blockIdx.y) + 
                (threadIdx.z + blockDim.z * blockIdx.z);

                // printf("index is %d\\n", idx); 

                pycuda::complex<float> j(0, 1);
                pycuda::complex<float> tmp(0, 0);

                for (int i=0; i < (sz) ; i++){
                    tmp += exp(j * (grid_x[idx] * fourier_pts_x[i] + grid_y[idx] * fourier_pts_y[i] + grid_z[idx] * fourier_pts_z[i])) * vol_f[i];
                }
                res[idx] = tmp;
            }
            """)

        bdim = (32, 32, 1)

        gridm = ((sz * sz * sz) / bdim[0] + ((sz * sz * sz) % bdim[0] > 0), 1)

        sz = np.uint32(sz)

        func = mod.get_function("adft3")
        func(cuda.In(vol_f), cuda.In(fourier_pts_x),
             cuda.In(fourier_pts_y), cuda.In(fourier_pts_z), cuda.In(grid_x), cuda.In(grid_y), cuda.In(grid_z), sz,
             cuda.Out(res),
             block=bdim, grid=gridm)
        return res.reshape([sz, sz, sz]), 0



if __name__ == "__main__":
    pass