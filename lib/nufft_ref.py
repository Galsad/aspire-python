import numpy as np
import nufft_cims


def calculate_kernel(omega, b, m, q, n):
    mu = np.round(omega * m)
    P = np.zeros([n, q + 1]).astype(np.complex64)

    # creating kernel function
    for i in range(-q / 2, q / 2 + 1):
        P[:, i + q / 2] = (np.exp(
            -(m * omega - (mu + i)) ** 2 / (4 * b))) / (
                              2 * np.sqrt(b * np.pi))

    return P


def kernel_nufft_1d(alpha, omega, M, precision='single'):
    '''
    reference code for GPU nufft
    :param alpha:
    :param omega:
    :param M:
    :param precision:
    :return:
    '''
    if precision == 'single':
        b = 0.5993
        m = 2
        q = 10

    elif precision == 'double':
        b = 1.5629
        m = 2
        q = 28

    else:
        raise "precision is not legall!"
    n = len(alpha)
    mu = np.round(omega * m)

    # P = calculate_kernel(omega, b, m, q, n)

    tau = np.zeros(m * M).astype(np.complex64)
    offset1 = np.ceil((m * M - 1) / 2.)

    # calculating tau
    for k in range(n):
        for j in range(-q / 2, q / 2 + 1):
            idx = mu[k] + j
            idx = ((idx + offset1) % (M * m))
            # tau[int(idx)] = tau[int(idx)] + P[k, j+q/2] * alpha[k]
            tau[int(idx)] = tau[int(idx)] + ((np.exp( -(m * omega[k] - (mu[k] + j)) ** 2 / (4 * b))) / ( 2 * np.sqrt(b * np.pi))) * alpha[k]

    T = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(tau)))
    T = T * len(T)

    low_idx_M = int(-np.ceil((M - 1) / 2.))
    high_idx_M = int(np.floor((M - 1) / 2.)) + 1
    idx = np.arange(low_idx_M, high_idx_M)
    E = np.exp(b * (2 * np.pi * idx / (m * M)) ** 2)
    E = E.flatten(order='F')
    offset2 = offset1 + low_idx_M
    f = T[int(offset2):int(offset2 + M)] * E

    return f


def kernel_nufft_2d(alpha, omega, M, precision='single'):
    '''
     reference code for GPU nufft
     :param alpha:
     :param omega: 2 x n array of sampling points
     :param M:
     :param precision:
     :return:
     '''
    if precision == 'single':
        b = 0.5993
        m = 2
        q = 10

    elif precision == 'double':
        b = 1.5629
        m = 2
        q = 28

    n = len(alpha)
    mu = np.round(omega * m)

    tau = np.zeros([m * M, m * M]).astype(np.complex64)
    offset1 = np.ceil((m * M - 1) / 2.)

    for k in range(n):
        for j1 in range(-q/2, q/2 + 1):
            for j2 in range(-q/2, q/2 + 1):
                idx = mu[k] + (j1, j2)
                idx = (idx + offset1) % (M*m)

                tmp = -((m*omega[k, 0] - (mu[k, 0] + j1))**2 + (m*omega[k, 1] - (mu[k, 1] + j2))**2) / (4*b)
                tmp2 = (np.exp(tmp) / (4 * b * np.pi)) * alpha[k]

                tau[int(idx[0]), int(idx[1])] = tau[int(idx[0]), int(idx[1])] + tmp2

    T = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(tau)))

    T = T * len(T) *len(T)


    low_idx_M = -np.ceil((M - 1) / 2.)
    high_idx_M = np.floor((M - 1) / 2.) + 1
    idx = np.arange(low_idx_M, high_idx_M)
    E = np.exp(b * (2. * np.pi * idx / (m * M)) ** 2);
    E = E.flatten(order='F')
    E = np.outer(E ,E)

    offset2 = offset1 + low_idx_M

    f = T[int(offset2):int(offset2) + M , int(offset2) : int(offset2) + M] * E
    return f

if __name__ == "__main__":

    ### testing 1D ###
    # n = 20

    # alpha = np.arange(-n / 2, n / 2) / float(n)
    # alpha = np.random.uniform(-np.pi, np.pi, n)
    # omega = np.arange(-n / 2, n / 2)

    # ret = kernel_nufft_1d(alpha, omega, n)

    ### testing 2D ###
    n = 100

    alpha = np.arange(-n/2, n/2) / float(n)
    #alpha = np.random.uniform(-np.pi, np.pi, n)
    omega_x = np.arange(-n/2, n/2)
    omega_y = np.arange(-n/2, n/2)

    omega = np.array([omega_x, omega_y]).transpose()


    ret = kernel_nufft_2d(alpha, omega, n)

    print ret
