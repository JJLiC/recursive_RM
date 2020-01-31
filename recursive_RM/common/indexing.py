import numpy as np
from itertools import product
from functools import lru_cache


@lru_cache()
def enum_subspace(m, d):
    ''' enumerate d-dimensional subspace index over m variables
        using integer RREF index

        For example, subspaces with basis 
            [[1, 0, 1], [0, 1, 0]]
        will be represented as 
            [5, 2]

    Arguments:
        m -- number of variables
        d -- subspace dimension

    Returns:
        Bs -- set of subspaces with integer RREF index
    '''
    if d == 0:
        # zero-dimensional trivial subspace
        Bs = [[]]
    elif d == m:
        # full standard basis for the whole space
        Bs = [[1 << i for i in range(m)[::-1]]]
    else:
        comb = np.array(list(product([0, 1], repeat=m - d)))
        # subspaces orthogonal to e_1
        Bs = enum_subspace(m - 1, d)[:]
        # subspaces not orthogonal to e_1
        for B in enum_subspace(m - 1, d - 1):
            # position of first ones
            i_B = set(np.log2(B).astype(int))
            # rest positions
            rest = list(set(range(m - 1)) - i_B)
            # all possible tails of the new row
            W = np.zeros((1 << (m - d), m - 1), dtype=int)
            W[:, rest] = comb
            # convert vector back to integer
            tails = W @ (1 << np.arange(m - 1))
            Bs += [[(1 << (m - 1)) + x] + B for x in tails]
    return Bs


@lru_cache()
def projection_index(m, d, filter_func=lambda B: True):
    ''' generate projection index for d-dimensional subspaces
        over m variables that pass the filter

        For example, a filter that checks all basis are 
        standard basis looks like
            filter_func=lambda B: (1 << np.log2(B).astype(int) == B).all()

    Arguments:
        m -- number of variables
        d -- subspace dimension

    Keyword Arguments:
        filter_func -- filter over subspaces (default: {lambda B: True})

    Returns:
        Bs -- subspaces passed filter, the number of them is n_B
        proj_idx -- n_B x 2^d x 2^(m - d)
                    llr_proj = CNOP(llr, axis=-2)
        B_idx, s_idx, t_idx -- all are n_B x 2^m
                    msg_V2C[(B_idx, s_idx, t_idx)] = llr + 
                        weight * (VNOP(msg_C2V, axis=-2) - msg_C2V)
    '''
    # Generate all d-dimensional subspace of F_2^m and filter
    Bs_int = [B for B in enum_subspace(m, d) if filter_func(B)]
    # For each subspace convert to basis vectors, MSB first
    h = 1 << np.arange(m)[None, None, ::-1]
    Bs = np.array(Bs_int)[..., None] // h % 2
    # Construct projection-puncturing system for each subspace
    PPS, I = [None] * len(Bs), np.eye(m, dtype=int)
    for b, B in enumerate(Bs):
        W, mask = B, np.array([False] * m)
        mask[W.argmax(axis=1)] = True
        U, V = I[mask], I[~mask]
        V_star = V
        # assert np.allclose(U @ V.T % 2, 0)
        # assert np.allclose(U @ W.T % 2, np.eye(d))
        # assert np.allclose(V_star @ V.T % 2, np.eye(m - d))
        PPS[b] = (U, V, W, V_star @ (I - W.T @ U) % 2)
    # Compute projection indices and aggregation indices
    hz, hs, ht = [1 << np.arange(k)[::-1] for k in [m, d, m - d]]
    Z = np.array(list(product([0, 1], repeat=m)))
    S = np.array(list(product([0, 1], repeat=d)))
    T = np.array(list(product([0, 1], repeat=m - d)))
    proj_idx, B_idx, s_idx, t_idx = [[None] * len(PPS) for _ in range(4)]
    for b, (U, V, W, Y) in enumerate(PPS):
        # z = W^T s + V^T t
        evalu_index = (S @ W % 2)[:, None, :]
        coset_index = (T @ V % 2)[None, ...]
        proj_idx_B = (evalu_index ^ coset_index) @ hz
        # s = U z,  t = V^* (I - W^T U) z
        s_idx_B, t_idx_B = Z @ U.T % 2 @ hs, Z @ Y.T % 2 @ ht
        # if (proj_idx_B[0] == np.arange(1 << (m - d))).all():
        #     print(U)
        #     print(proj_idx_B)
        # assert d == 1 or np.allclose(np.bitwise_xor.reduce(proj_idx_B, axis=0), 0)
        # assert all(proj_idx_B[s, t] == v for v, s, t in zip(range(1 << m), s_idx_B, t_idx_B))
        proj_idx[b], B_idx[b] = proj_idx_B, np.full(1 << m, b)
        s_idx[b], t_idx[b] = s_idx_B, t_idx_B
    return [np.array(arr) for arr in [proj_idx, B_idx, s_idx, t_idx]]


@lru_cache()
def puncturing_index(m, d, filter_func=lambda B: True):
    ''' generate puncturing index for d-dimensional subspaces
        over m variables that pass the filter

        For example, a filter that checks all basis are 
        standard basis looks like
            filter_func=lambda B: (np.log2(B) == B).all()

    Arguments:
        m -- number of variables
        d -- subspace dimension

    Keyword Arguments:
        filter_func -- filter over subspaces (default: {lambda B: True})

    Returns:
        Bs -- subspaces passed filter, the number of them is n_B
        punc_idx -- n_B x 2^d x 2^(m - d)
                    llr_punc = llr[punc_idx]
        B_idx, s_idx, t_idx -- all are n_B x 2^m
                    msg_V2C[(B_idx, s_idx, t_idx)] = llr + 
                        weight * (VNOP(msg_C2V, axis=-2) - msg_C2V)
    '''
    # Generate all d-dimensional subspace of F_2^m and filter
    Bs_int = [B for B in enum_subspace(m, d) if filter_func(B)]
    # For each subspace convert to basis vectors, MSB first
    h = 1 << np.arange(m)[None, None, ::-1]
    Bs = np.array(Bs_int)[..., None] // h % 2
    # Construct projection-puncturing system for each subspace
    PPS, I = [None] * len(Bs), np.eye(m, dtype=int)
    for b, B in enumerate(Bs):
        U, mask = B, np.array([False] * m)
        mask[U.argmax(axis=1)] = True
        P, Q = U[:, ~mask], np.vstack([I[mask], I[~mask]])
        V, W, V_star = (P.T @ Q[:d] + Q[d:]) % 2, Q[:d], Q[d:]
        # assert np.allclose(U @ V.T % 2, 0)
        # assert np.allclose(U @ W.T % 2, np.eye(d))
        # assert np.allclose(V_star @ V.T % 2, np.eye(m - d))
        PPS[b] = (U, V, W, V_star @ (I - W.T @ U) % 2)
    # Compute projection indices and aggregation indices
    hz, hs, ht = [1 << np.arange(k)[::-1] for k in [m, d, m - d]]
    Z = np.array(list(product([0, 1], repeat=m)))
    S = np.array(list(product([0, 1], repeat=d)))
    T = np.array(list(product([0, 1], repeat=m - d)))
    punc_idx, B_idx, s_idx, t_idx = [[None] * len(PPS) for _ in range(4)]
    for b, (U, V, W, Y) in enumerate(PPS):
        # z = W^T s + V^T t
        evalu_index = (S @ W % 2)[:, None, :]
        coset_index = (T @ V % 2)[None, ...]
        punc_idx_B = (evalu_index ^ coset_index) @ hz
        # if (punc_idx_B[0] == np.arange(1 << (m - d))).all():
        #     print(U)
        #     print(punc_idx_B)
        # s = U z,  t = V^* (I - W^T U) z
        s_idx_B, t_idx_B = Z @ U.T % 2 @ hs, Z @ Y.T % 2 @ ht
        # assert d == 1 or np.allclose(np.bitwise_xor.reduce(punc_idx_B, axis=0), 0)
        # assert all(punc_idx_B[s, t] == v for v, s, t in zip(range(1 << m), s_idx_B, t_idx_B))
        punc_idx[b], B_idx[b] = punc_idx_B, np.full(1 << m, b)
        s_idx[b], t_idx[b] = s_idx_B, t_idx_B
    return [np.array(arr) for arr in [punc_idx, B_idx, s_idx, t_idx]]
