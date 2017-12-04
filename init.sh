#!/bin/sh
WORKINGDIR=`pwd`
export WORKINGDIR=`pwd`
export NUFFTDIR=${WORKINGDIR}

export PATH=${PATH}:${NUFFTDIR}
export PYTHONPATH=${PYTHONPATH}:${NUFFTDIR}

export PATH=${PATH}:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
export CUDA_ROOT=/usr/local/cuda-8.0/bin

echo "DONE!"

export PATH=${PATH}:${NUFFTDIR}/lib
export PATH=${PATH}:${NUFFTDIR}/extern
export PATH=${PATH}:${NUFFTDIR}/tests
export PATH=${PATH}:${NUFFTDIR}/programs

export PYTHONPATH=${PYTHONPATH}:${NUFFTDIR}/lib
export PYTHONPATH=${PYTHONPATH}:${NUFFTDIR}/extern
export PYTHONPATH=${PYTHONPATH}:${NUFFTDIR}/tests
export PYTHONPATH=${PYTHONPATH}:${NUFFTDIR}/programs