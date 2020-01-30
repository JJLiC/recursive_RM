import numpy as np


def fwht(x):
    ''' fast Walsh-Hadamard transform on the last dimension
    Arguments:
        x -- input vector
    Returns:
        a -- output vector
    '''
    h, n = 1, x.shape[-1]
    a, idx = x.copy(), np.arange(x.shape[-1], dtype=int)
    while h < n:
        mask = (idx & h) == 0
        x, y = a[..., mask], a[..., ~mask]
        a[..., mask], a[..., ~mask] = x + y, x - y
        h *= 2
    return a


def fwht_Z2(x0, x1):
    '''binary fast Walsh-Hadamard transform on the last dimension
    Arguments:
        x0 -- input vector corresponding to 0
        x1 -- input vector corresponding to 1
    Returns:
        a -- output vector
    '''
    h, n = 1, x0.shape[-1]
    b0, b1 = x0.copy(), x1.copy()
    d0, d1 = np.zeros_like(b0), np.zeros_like(b1)
    while h < n:
        d0[..., :n // 2] = b0[..., ::2] ^ b0[..., 1::2]
        d0[..., n // 2:] = b0[..., ::2] ^ b1[..., 1::2]
        d1[..., :n // 2] = b1[..., ::2] ^ b1[..., 1::2]
        d1[..., n // 2:] = b1[..., ::2] ^ b0[..., 1::2]
        b0, b1, d0, d1 = d0, d1, b0, b1
        h *= 2
    a = np.concatenate([b0, b1], axis=-1)
    return a
