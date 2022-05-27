from math import ceil
import numpy
import time

from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
import warnings

from MCOP import MCOP
from MCOP_GPU import MCOP_GPU

import pycuda.autoinit
import pycuda.gpuarray
import pycuda.curandom

#Podatci 
N_PATHS = numpy.int64(11920)
N_STEP = numpy.int64(365)
N_NORMALS = N_PATHS * N_STEP

T = numpy.float32(1.0)
K = numpy.float32(110.0)
B = numpy.float32(100.0)
S0 = numpy.float32(120.0)
sigma = numpy.float32(0.35)
mu = numpy.float32(0.1)
r = numpy.float32(0.05)

#Izrada arraya
s = numpy.zeros((N_PATHS), dtype=numpy.float32)
d_s = numpy.zeros((N_PATHS), dtype=numpy.float32)
d_normals = pycuda.gpuarray.GPUArray((N_NORMALS), dtype=numpy.float32)
normals = numpy.zeros((N_NORMALS), dtype=numpy.float32)

#Izrada random brojeva pomocu GPU
rng = pycuda.curandom.MRG32k3aRandomNumberGenerator()

rng.fill_uniform(d_normals)

#CPU
normals = d_normals.get()

time1 = time.time()
sum = MCOP(N_PATHS, N_STEP, mu, sigma, normals, K, r, T, B, S0)

sum = sum / N_PATHS
time2 = time.time()

print("CPU odgovor: " + str(sum))
print("Vrijeme na CPU: " + str(time2 - time1) + " sec")

#GPU

time1 = time.time()
grid = ceil(N_PATHS / 1024)
block = 1024
cuda.select_device(0)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
MCOP_GPU[grid, block](N_PATHS, N_STEP, mu, sigma, normals, K, r, T, B, S0, d_s)

gpu_sum = 0

for i in range(N_PATHS):
    gpu_sum = gpu_sum + d_s[i]

gpu_sum = gpu_sum / N_PATHS
time2 = time.time()

print("\n\nGPU odgovor: " + str(gpu_sum))
print("Vrijeme na GPU: " + str(time2 - time1) + " sec")


