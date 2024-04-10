CUTLASS_FORK=~/cutlass-fork
CUDA=/opt/hpc_software/compilers/nvidia/cuda-12.3
DPCPP=~/sycl_linux_20240327
$DPCPP/bin/clang++ -fsycl -DCUTLASS_ENABLE_SYCL sycl.cpp -I ../../include/ -I $CUTLASS_FORK/include/ -I $CUDA/include -fcolor-diagnostics 
