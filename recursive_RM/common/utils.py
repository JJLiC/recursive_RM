import numpy as np


def rref(A):
    ''' Compute Reduced Row-Echelon Form of a matrix
    Arguments:
        A -- input matrix
    Returns:
        R -- R * A = rref(A)
        Y -- rref(A)
        rank -- rank of A
    '''
    (m, n), j = A.shape, 0
    Ar, rank = np.hstack([A, np.eye(m, dtype=int)]), 0
    for i in range(min(m, n)):
        # Find value and index of non-zero element in the remainder of column i.
        while j < n:
            temp = np.where(Ar[i:, j] != 0)[0]
            if len(temp) == 0:
                # If the lower half of j-th row is all-zero, check next column
                j += 1
            else:
                # Swap i-th and k-th rows
                k, rank = temp[0] + i, rank + 1
                if i != k:
                    Ar[[i, k], j:] = Ar[[k, i], j:]
                # Save the right hand side of the pivot row
                pivot = Ar[i, j]
                row = Ar[i, j:].reshape((1, -1)) / pivot
                col = np.hstack([Ar[:i, j], [0], Ar[i + 1:, j]]).reshape((-1, 1))
                Ar[:, j:] = (Ar[:, j:] - col * row) % 2
                Ar[i, j:] = Ar[i, j:] / pivot % 2
                break
        j += 1
    R, Y = Ar[:, :n], Ar[:, n:]
    return R, Y, rank


def systemize(G):
    ''' Systemize generator matrix G
    Arguments:
        G -- input generator matrix
    Returns:
        Gsys -- systematic form of G
        perm -- Gsys = rref(G) with column permutation perm
        inv_perm -- inverse permutation of perm
    '''
    Grref = rref(G)[0]
    col_sum = Grref.sum(axis=0)
    perm = np.concatenate([np.where(col_sum == 1)[0], np.where(col_sum > 1)[0]])
    Gsys = Grref[:, perm]
    inv_perm = perm.argsort()
    return Gsys, perm, inv_perm


def AWGN_output(G, rho_dB, rng=None):
    ''' Transmit codewords of G over AWGN channel
    Arguments:
        G -- generator matrix
        rho_dB -- channel SNR in dB
    Keyword Arguments:
        rng -- random number generator (default: {None})
    Returns:
        u -- information bits
        c -- transmitted codeword
        y -- channel output
        llr -- LLR of channel output
    '''
    m = int(np.log2(G.shape[1]))
    R, rho = G.shape[0] / G.shape[1], 10**(rho_dB / 10)
    std = 1 / np.sqrt(2 * R * rho)
    if rng:
        u = rng.randint(0, 2, size=G.shape[0])
        c = u @ G % 2
        y = (-1)**c + rng.randn(2**m) * std
    else:
        u = np.random.randint(0, 2, size=G.shape[0])
        c = u @ G % 2
        y = (-1)**c + np.random.randn(2**m) * std
    llr = 4 * R * rho * y
    return u, c, y, llr
