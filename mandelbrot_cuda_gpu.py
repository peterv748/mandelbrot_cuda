# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:44:19 2018

@author: peter
"""

import numpy as np
from numpy import NaN
from numba import cuda
import matplotlib.pyplot as plt
import time

@cuda.jit(device=True) 
def mandelbrot(creal,cimag,maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal       
    return maxiter




@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, X_size, Y_size, image,  iters):

  
  pixel_size_x = (max_x - min_x) / X_size
  pixel_size_y = (max_y - min_y) / Y_size
  
  startX, startY = cuda.grid(2)


  startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
  startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;

  for x in range(startX, X_size, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, Y_size, gridY):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandelbrot(real, imag, iters)



#initialisations of constants
X_size = 4096
Y_size = 4096
blockdim = (32, 32)
griddim = (128,128)
xmin = -2
xmax = 0.5
ymin = -1
ymax = 1
maxiter = 300
image = np.zeros((Y_size, X_size), dtype = np.uint32)


# start calculations

d_image = cuda.to_device(image)
start = time.time()
mandel_kernel[griddim, blockdim](xmin,xmax, ymin, ymax, X_size, Y_size, d_image, maxiter) 
dt = time.time() - start
d_image.copy_to_host(image)



#plot image in window
plt.imshow(image, cmap = plt.cm.prism, interpolation = 'none', extent = (xmin, xmax, ymin, ymax))
plt.xlabel("Re(c), using cuda nvidia processing time: %f s" % dt)
plt.ylabel("Im(c), maxiter 300")
plt.title("mandelbrot set, image size (x,y): 4096 x 4096 pixels")
plt.savefig("mandelbrot_python_optimize_cuda_gpu.png")
plt.show()
plt.close()
