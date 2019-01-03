
import numpy as np
from numpy import NaN
from numba import jit
import matplotlib.pyplot as plt
import time

@jit 
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

@jit
def mandelbrot_set(xmin,xmax,ymin,ymax,X_size, Y_size, maxiter):
        stepsize_x = (xmax - xmin)/X_size
        stepsize_y = (ymax - ymin)/Y_size
        X = np.arange(xmin, xmax, stepsize_x)
        Y = np.arange(ymin, ymax, stepsize_y)
        Z = np.zeros((len(Y), len(X)))
        for iy, y in enumerate(Y):
            for ix, x in enumerate(X):
                Z[iy,ix] = mandelbrot(x,y, maxiter)

        return (Z)

#initialisation of constants   
xmin = -2
xmax = .5
ymin = -1
ymax = 1
X_size = 4096
Y_size = 4096
maxiter = 300


# start calculation
start = time.time()
Z = mandelbrot_set(xmin, xmax, ymin, ymax, X_size, Y_size, maxiter)
dt = time.time() - start


#plot image in window
plt.imshow(Z, cmap = plt.cm.prism, interpolation = 'none', extent = (xmin, xmax, ymin, ymax))
plt.xlabel("Re(c), using numba jit compiler time: %f s" % dt)
plt.ylabel("Im(c), max iter =300")
plt.title( "mandelbrot set, image size (x,y): 4096 x 4096 pixels")
plt.savefig("mandelbrot_python_optimaze_cuda_jit.png")
plt.show()
plt.close()
