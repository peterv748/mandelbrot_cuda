# mandelbrot_cuda
mandelbrot implementation using Python, numpy, numba and a GTX1070 mobile graphics GPU and CUDA version 10
it is necessary to install, using pip or any other package installer, numpy, numba cuda library, matplotlib.pyplot. Also needed is the installation of the CUDA package from NVIDIA
the code stores the mandelbrot calculation in an array of 4096x4096 points, and displays this array using matplotlib functions.
included are three png files: 
- showing processing time without any optimization 
- showing processing time usig numba jit optimazition  
- showing processing time usig a NVIDIA GTX1070 mobile, using CUDA version 10 API (this code)

All is run on a HP OMEN laptop, having 16GB memory and a Intel Core i8 8750, with 6 cores
