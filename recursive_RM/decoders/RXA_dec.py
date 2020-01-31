import numpy as np
from numpy.linalg import norm
from recursive_RM.common.fwht import fwht, fwht_Z2
from recursive_RM.common.indexing import puncturing_index


def ext_Hamming_decode_soft(L):
    ''' soft bit-MAP decoder for RM(m - 2, m) on the last axis
    Arguments:
        L -- input LLR
    Returns:
        L_out -- output LLR
    '''
    def fix(x):
        y, ind = x, np.where(x == 0)
        val = np.random.choice([-eps, eps], size=ind[0].size)
        y[ind] = val
        return y

    eps = 1e-15
    n = L.shape[-1]
    rho = fix(np.tanh(L / 2))
    beta0, beta1 = (L < 0).astype(int), np.zeros(L.shape, dtype=int)
    tau = np.log(np.maximum(abs(rho), eps))
    s, delta = fwht(tau), fwht_Z2(beta0, beta1)
    t = (s[..., [0]] + np.concatenate([s, -s], axis=-1)) / 2
    t = np.clip(t, -30, 30)
    h = (-1) ** delta * np.exp(t)
    r, q = fwht(h[..., :n] - h[..., n:]), h.sum(axis=-1, keepdims=True)
    w0, w1 = (q - r) / 2, (q + r) / 2
    z = (w0 * rho + w1 / rho) / fix(w0 + w1)
    z = np.clip(z, -1 + eps, 1 - eps)
    L_out = 2 * np.arctanh(z)
    return L_out


def RXA(llr, r, m, weight=None, damp=1, t_max=15, theta=0.01, root=True):
    ''' Recursive Puncture-Aggregation BP algorithm
    Arguments:
        llr -- input LLR
        r -- degree of Reed-Muller code
        m -- number of variables of Reed-Muller code
    Keyword Arguments:
        weight -- scale factor for BP messages (default: number of subspace)
        damp -- damping coefficient (default: 1)
        t_max -- maximum iteration number (default: 15)
        theta -- relative threshold for stopping criterion (default: 0.01)
        root -- if this is the root function call (default: True)
    Returns:
        x_hat -- if root, return decoded codeword, else return LLR
    '''
    if r + 2 == m:
        # Avoid zeros in subcode LLR
        llr = np.maximum(abs(llr), 1e-4) * np.sign(llr)
        # Decode base extended Hamming subcode
        llr_hat = ext_Hamming_decode_soft(llr)
    else:
        # Compute index and message shapes
        punc_idx, *aggr_idx = puncturing_index(m, 1)
        nd_idx = np.ogrid[tuple(slice(d) for d in llr.shape[:-1] + (1, 1))][:-2]
        n_B, aggr_idx = punc_idx.shape[0], tuple(nd_idx) + tuple(aggr_idx)
        fwd_msg_shape = llr.shape[:-1] + (n_B, 1 << 1, 1 << (m - 1))
        bkd_msg_shape = llr.shape[:-1:2] + (n_B, 1 << m)

        # Reshape input LLR to fit high-dimensional array
        llr = llr[..., None, :]
        # Set weight in aggregation step
        weight = weight or 1 / n_B

        # Initialize BP message and result of previous iteration
        fwd_msg = np.zeros(fwd_msg_shape)
        fwd_msg[aggr_idx] = llr
        subcode_llr_hat_old = fwd_msg.copy()
        bkd_msg = np.zeros(bkd_msg_shape)
        aggr_msg_old = np.tile(llr, (n_B, 1))
        llr_hat_old = np.zeros_like(llr)
        x_hat_old = np.zeros_like(llr, dtype=int)

        for t in range(t_max):
            # Avoid zeros in subcode LLR
            fwd_msg = np.maximum(abs(fwd_msg), 1e-4) * np.sign(fwd_msg)

            # Puncture and decode subcodes
            subcode_llr_hat = RXA(fwd_msg, r, m - 1, weight, damp, t_max, theta, root=False)
            # Apply damping if coefficient is not 1
            if damp != 1:
                subcode_llr_hat *= damp
                subcode_llr_hat += (1 - damp) * subcode_llr_hat_old
            # Clip value for numerical stability
            subcode_llr_hat = np.clip(subcode_llr_hat, -30, 30)
            subcode_llr_hat_old = subcode_llr_hat
            # Assign subcode LLR to backward messages
            bkd_msg = subcode_llr_hat[aggr_idx]

            # Aggregation and apply BP
            aggr_sum = bkd_msg.sum(axis=-2, keepdims=True)
            aggr_msg = llr + weight * (aggr_sum - bkd_msg)
            # Apply damping if coefficient is not 1
            if damp != 1:
                aggr_msg *= damp
                aggr_msg += (1 - damp) * aggr_msg_old
            # Clip value for numerical stability
            aggr_msg = np.clip(aggr_msg, -30, 30)
            aggr_msg_old = aggr_msg
            # Assign backward messages to subcode LLR
            fwd_msg[aggr_idx] = aggr_msg

            # Marginalization and stopping criterion
            llr_hat = weight * aggr_sum
            x_hat = llr_hat < 0
            if (x_hat == x_hat_old).all() and norm(llr_hat - llr_hat_old) < theta * norm(llr_hat):
                break
            # Update result of last iteration
            llr_hat_old, x_hat_old = llr_hat, x_hat
    return x_hat.squeeze().astype(int) if root else llr_hat.squeeze()


def CXA(llr, r, m, weight=None, damp=1, t_max=15, theta=0.01):
    ''' Collapsed Recursive Puncture-Aggregation BP algorithm
    Arguments:
        llr -- input LLR
        r -- degree of Reed-Muller code
        m -- number of variables of Reed-Muller code
    Keyword Arguments:
        weight -- scale factor for BP messages (default: number of subspace)
        damp -- damping coefficient (default: 1)
        t_max -- maximum iteration number (default: 15)
        theta -- relative threshold for stopping criterion (default: 0.01)
    Returns:
        x_hat -- decoded codeword
    '''
    # Compute index and message shapes
    d = m - r - 2
    punc_idx, *aggr_idx = puncturing_index(m, d)
    n_B, aggr_idx = punc_idx.shape[0], tuple(aggr_idx)
    fwd_msg_shape = (n_B, 1 << d, 1 << (m - d))
    bkd_msg_shape = (n_B, 1 << m)

    # Set weight in aggregation step
    weight = weight or 1 / n_B

    # Initialize BP message and result of previous iteration
    fwd_msg = np.zeros(fwd_msg_shape)
    fwd_msg[aggr_idx] = llr
    subcode_llr_hat_old = fwd_msg.copy()
    bkd_msg = np.zeros(bkd_msg_shape)
    aggr_msg_old = np.tile(llr, (n_B, 1))
    llr_hat_old = np.zeros_like(llr)
    x_hat_old = np.zeros_like(llr, dtype=int)

    for t in range(t_max):
        # Avoid zeros in subcode LLR
        fwd_msg = np.maximum(abs(fwd_msg), 1e-4) * np.sign(fwd_msg)

        # Puncture to sub-code and decode extended Hamming
        subcode_llr_hat = ext_Hamming_decode_soft(fwd_msg)
        # Apply damping if coefficient is not 1
        if damp != 1:
            subcode_llr_hat *= damp
            subcode_llr_hat += (1 - damp) * subcode_llr_hat_old
        # Clip value for numerical stability
        subcode_llr_hat = np.clip(subcode_llr_hat, -30, 30)
        subcode_llr_hat_old = subcode_llr_hat
        # Assign subcode LLR to backward messages
        bkd_msg = subcode_llr_hat[aggr_idx]

        # Aggregation and apply BP
        aggr_sum = bkd_msg.sum(axis=-2)
        aggr_msg = llr + weight * (aggr_sum - bkd_msg)
        # Apply damping if coefficient is not 1
        if damp != 1:
            aggr_msg *= damp
            aggr_msg += (1 - damp) * aggr_msg_old
        # Clip value for numerical stability
        aggr_msg = np.clip(aggr_msg, -30, 30)
        aggr_msg_old = aggr_msg
        # Assign backward messages to subcode LLR
        fwd_msg[aggr_idx] = aggr_msg

        # Marginalization and stopping criterion
        llr_hat = weight * aggr_sum
        x_hat = llr_hat < 0
        if (x_hat == x_hat_old).all() and norm(llr_hat - llr_hat_old) < theta * norm(llr_hat):
            break
        # Update result of last iteration
        llr_hat_old, x_hat_old = llr_hat, x_hat
    return x_hat.astype(int)
