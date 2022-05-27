import numpy

def MCOP(N_PATHS, N_STEPS, mu, sigma, normals, K, r, T, B, S0):
    tmp1 = mu * T / N_STEPS
    tmp2 = numpy.exp(-r * T)
    tmp3 = numpy.sqrt(T / N_STEPS)
    running_avg = 0
    sum = 0

    for i in range(N_PATHS):
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
        
        sum = sum + (tmp2 * payoff)

    return sum

        