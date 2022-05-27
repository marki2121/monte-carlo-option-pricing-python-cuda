import math
from numba import cuda

@cuda.jit
def MCOP_GPU(N_PATHS, N_STEPS, mu, sigma, normals, K, r, T, B, S0, d_s):
    ii = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x

    tmp1 = mu * T / N_STEPS
    tmp2 = math.exp(-r * T)
    tmp3 = math.sqrt(T / N_STEPS)
    running_avg = 0

    for i in range(ii, N_PATHS, stride):
        s_curr = S0

        for n in range(N_STEPS):
            s_curr = s_curr + (tmp1 * s_curr + sigma * s_curr * tmp3 * normals[i + n * N_PATHS])
            running_avg = running_avg + 1.0/(n + 1.0) * (s_curr - running_avg)

            if(running_avg <= B):
                break
        
        if(running_avg > K):
            payoff = running_avg - K
        else:
            payoff = 0
        
        d_s[i] = tmp2 * payoff


        