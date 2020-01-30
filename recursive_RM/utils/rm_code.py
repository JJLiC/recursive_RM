import numpy as np
from itertools import product


def RM(r, m):
    ''' generator matrix Reed-Muller code
    Arguments:
        r -- degree of Reed-Muller code
        m -- number of variables of Reed-Muller code
    Returns:
        G -- generator matrix
    '''
    G, z = [], np.arange(2**m, dtype=int)
    basis = np.array([(z >> d) & 1 for d in range(m)[::-1]], dtype=int)
    for mask in product([False, True], repeat=m):
        if sum(mask) <= r:
            G.append(np.prod(basis[list(mask)], axis=0))
    G = np.vstack(G)
    return G


def rm1_codeword(u_0, u_star, m):
    ''' given information bits in the form of integer pair [u_0, u_star]
        return corresponding first-order Reed-Muller codewords
    Arguments:
        u_0 -- constant term in RM code
        u_star -- index of Hadamard rows
        m -- number of variables of Reed-Muller code
    Returns:
        cw -- corresponding first-order Reed-Muller codewords
    '''
    Z = u_star[..., None] & np.arange(2**m, dtype=int)
    cw = np.zeros_like(Z)
    for _ in range(m):
        cw, Z = cw ^ (Z & 1), Z >> 1
    cw = u_0[..., None] ^ cw
    return cw
