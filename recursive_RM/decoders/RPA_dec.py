import numpy as np
from numpy.linalg import norm
from recursive_RM.common.rm_code import rm1_codeword
from recursive_RM.common.fwht import fwht
from recursive_RM.common.indexing import projection_index


def fht_RM1_decode(L, m):
    ''' word-MAP decoder for first-order Reed-Muller code
    Arguments:
        L -- input LLR
        m -- number of variables
    Returns:
        c_hat -- decoded codeword
    '''
    L_hat = fwht(L)
    u_star = abs(L_hat).argmax(axis=-1)
    nd_idx = np.ogrid[tuple(slice(d) for d in L.shape[:-1])]
    u_0 = (L_hat[(*nd_idx, u_star)] <= 0).astype(int)
    c_hat = rm1_codeword(u_0, u_star, m)
    return c_hat


def fht_RM1_decode_soft(L, llr_clip=30):
    ''' soft bit-MAP decoder for RM(1, m)
    Arguments:
        L -- input LLR
    Keyword Arguments:
        llr_clip -- LLR clipping value {default: 30}
    Returns:
        L_out -- output LLR
    '''
    L_hat = np.clip(fwht(L) / 2, -llr_clip, llr_clip)
    L_temp = fwht(np.sinh(L_hat)) / np.cosh(L_hat).sum(axis=-1, keepdims=True)
    L_temp = np.clip(L_temp, -1 + 1e-15, 1 - 1e-15)
    L_out = 2 * np.arctanh(L_temp)
    return L_out


def RPA(llr, r, m, weight={}, damp=1, t_max=15, theta=0.01, llr_clip=30, base_dec='hard', root=True):
    ''' Recursive Puncture-Aggregation BP algorithm
    Arguments:
        llr -- input LLR
        r -- degree of Reed-Muller code
        m -- number of variables of Reed-Muller code
    Keyword Arguments:
        weight -- dict in form of [(r, m): scale]
                  scale factor for BP messages (default: {})
        damp -- damping coefficient (default: 1)
        t_max -- maximum iteration number (default: 15)
        theta -- relative threshold for stopping criterion (default: 0.01)
        llr_clip -- LLR clipping value {default: 30}
        base_dec -- base case decoding algorithm (default: 'hard')
        root -- if this is the root function call (default: True)
    Returns:
        x_hat -- if root, return decoded codeword, else return LLR
    '''
    if r == 1:
        if base_dec == 'hard':
            llr_hat = llr_clip * (-1) ** fht_RM1_decode(llr, m)
        else:
            llr_hat = fht_RM1_decode_soft(llr, llr_clip)
    else:
        # Compute index and message shapes
        proj_idx, *aggr_idx = projection_index(m, 1)
        nd_idx = np.ogrid[tuple(slice(d) for d in llr.shape[:-1] + (1, 1))][:-2]
        n_B, aggr_idx = proj_idx.shape[0], tuple(nd_idx) + tuple(aggr_idx)
        fwd_msg_shape = llr.shape[:-1] + (n_B, 1 << 1, 1 << (m - 1))
        bkd_msg_shape = llr.shape[:-1] + (n_B, 1 << m)

        # Reshape input LLR to fit high-dimensional array
        llr = llr[..., None, :]
        # Set scale in aggregation step
        scale = weight.get((r, m), 1 / n_B)

        # Initialize BP message and result of previous iteration
        fwd_msg = np.zeros(fwd_msg_shape)
        bkd_msg = np.zeros(bkd_msg_shape)
        fwd_msg[aggr_idx] = llr
        subcode_llr_hat_old = fwd_msg.copy()
        aggr_msg_old = np.tile(llr, (n_B, 1))
        llr_hat_old = np.zeros_like(llr)
        x_hat_old = np.zeros_like(llr, dtype=int)

        for t in range(t_max):
            # Avoid zeros in fwd_msg
            fwd_msg = np.maximum(abs(fwd_msg), 1e-4) * np.sign(fwd_msg)

            # Project to sub-code and decode first-order Reed-Muller
            temp = np.prod(np.tanh(fwd_msg / 2), axis=-2)
            # Clip value to avoid numerical issue of arctanh
            temp = np.clip(temp, -1 + 1e-15, 1 - 1e-15)
            subcode_llr = 2 * np.arctanh(temp)
            subcode_llr_hat = RPA(subcode_llr, r - 1, m - 1, weight, damp, t_max, theta, llr_clip, base_dec, root=False)
            # subcode_llr_hat = subcode_llr_hat - subcode_llr
            # Apply damping if coefficient is not 1
            if damp != 1:
                subcode_llr_hat *= damp
                subcode_llr_hat += (1 - damp) * subcode_llr_hat_old
            # Clip value for numerical stability
            subcode_llr_hat = np.clip(subcode_llr_hat, -llr_clip, llr_clip)
            subcode_llr_hat_old = subcode_llr_hat

            # Apply backward BP
            temp = np.tanh(fwd_msg / 2)
            fwd_prod = np.prod(temp, axis=-2, keepdims=True)
            temp = np.tanh(subcode_llr_hat[..., None, :] / 2) * fwd_prod / temp
            # Clip value to avoid numerical issue of arctanh
            temp = np.clip(temp, -1 + 1e-15, 1 - 1e-15)
            # Assign to backward messages
            bkd_msg = (2 * np.arctanh(temp))[aggr_idx]

            # Aggregation and apply forward BP
            aggr_sum = bkd_msg.sum(axis=-2, keepdims=True)
            if base_dec == 'hard':
                aggr_msg = scale * (aggr_sum - bkd_msg)
            else:
                aggr_msg = llr + scale * (aggr_sum - bkd_msg)
            # Apply damping if coefficient is not 1
            if damp != 1:
                aggr_msg *= damp
                aggr_msg += (1 - damp) * aggr_msg_old
            # Clip value for numerical stability
            aggr_msg = np.clip(aggr_msg, -llr_clip, llr_clip)
            aggr_msg_old = aggr_msg
            # Assign backward messages to subcode LLR
            fwd_msg[aggr_idx] = aggr_msg

            # Marginalization and stopping criterion
            llr_hat = scale * aggr_sum
            x_hat = llr_hat < 0
            if (x_hat == x_hat_old).all() and norm(llr_hat - llr_hat_old) < theta * norm(llr_hat):
                break
            # Update result of last iteration
            llr_hat_old, x_hat_old = llr_hat, x_hat
    return x_hat.squeeze().astype(int) if root else llr_clip * (-1) ** (llr_hat < 0).squeeze()


def CPA(llr, r, m, weight={}, damp=1, t_max=15, theta=0.01, llr_clip=30, base_dec='hard'):
    ''' Collapsed Recursive Projection-Aggregation BP algorithm
    Arguments:
        llr -- input LLR
        r -- degree of Reed-Muller code
        m -- number of variables of Reed-Muller code
    Keyword Arguments:
        weight -- dict in form of [(r, m): scale]
                  scale factor for BP messages (default: {})
        damp -- damping coefficient (default: 1)
        t_max -- maximum iteration number (default: 15)
        theta -- relative threshold for stopping criterion (default: 0.01)
        llr_clip -- LLR clipping value {default: 30}
        base_dec -- base case decoding algorithm (default: 'hard')
                    if hard, use intrinsic, otherwise use extrinsic
    Returns:
        x_hat -- decoded codeword
    '''
    # Compute index and message shapes
    d = r - 1
    proj_idx, *aggr_idx = projection_index(m, d)
    n_B, aggr_idx = proj_idx.shape[0], tuple(aggr_idx)
    fwd_msg_shape = (n_B, 1 << d, 1 << (m - d))
    bkd_msg_shape = (n_B, 1 << m)

    # Set weight in aggregation step
    scale = weight.get((r, m), 1 / n_B)

    # Initialize BP message and result of previous iteration
    fwd_msg = np.zeros(fwd_msg_shape)
    fwd_msg[aggr_idx] = llr
    subcode_llr_hat_old = fwd_msg.copy()
    bkd_msg = np.zeros(bkd_msg_shape)
    aggr_msg_old = np.tile(llr, (n_B, 1))
    llr_hat_old = np.zeros_like(llr)
    x_hat_old = np.zeros_like(llr, dtype=int)

    for t in range(t_max):
        # Avoid zeros in fwd_msg
        fwd_msg = np.maximum(abs(fwd_msg), 1e-4) * np.sign(fwd_msg)

        # Project to sub-code and decode first-order Reed-Muller
        temp = np.prod(np.tanh(fwd_msg / 2), axis=-2, keepdims=True)
        # Clip value to avoid numerical issue of arctanh
        temp = np.clip(temp, -1 + 1e-15, 1 - 1e-15)
        subcode_llr = 2 * np.arctanh(temp)
        if base_dec == 'hard':
            subcode_llr_hat = llr_clip * (-1) ** fht_RM1_decode(subcode_llr, m - d)
        else:
            subcode_llr_hat = fht_RM1_decode_soft(subcode_llr, llr_clip)
        # subcode_llr_hat = subcode_llr_hat - subcode_llr
        # Apply damping if coefficient is not 1
        if damp != 1:
            subcode_llr_hat *= damp
            subcode_llr_hat += (1 - damp) * subcode_llr_hat_old
        # Clip value for numerical stability
        subcode_llr_hat = np.clip(subcode_llr_hat, -llr_clip, llr_clip)
        subcode_llr_hat_old = subcode_llr_hat

        # Apply backward BP
        temp = np.tanh(fwd_msg / 2)
        fwd_prod = np.prod(temp, axis=-2, keepdims=True)
        temp = np.tanh(subcode_llr_hat / 2) * fwd_prod / temp
        # Clip value to avoid numerical issue of arctanh
        temp = np.clip(temp, -1 + 1e-15, 1 - 1e-15)
        # Assign to backward messages
        bkd_msg = (2 * np.arctanh(temp))[aggr_idx]

        # Aggregation and apply forward BP
        aggr_sum = bkd_msg.sum(axis=-2)
        if base_dec == 'hard':
            aggr_msg = scale * (aggr_sum - bkd_msg)
        else:
            aggr_msg = llr + scale * (aggr_sum - bkd_msg)
        # Apply damping if coefficient is not 1
        if damp != 1:
            aggr_msg *= damp
            aggr_msg += (1 - damp) * aggr_msg_old
        # Clip value for numerical stability
        aggr_msg = np.clip(aggr_msg, -llr_clip, llr_clip)
        aggr_msg_old = aggr_msg
        # Assign backward messages to subcode LLR
        fwd_msg[aggr_idx] = aggr_msg

        # Marginalization and stopping criterion
        llr_hat = scale * aggr_sum
        x_hat = llr_hat < 0
        if (x_hat == x_hat_old).all() and norm(llr_hat - llr_hat_old) < theta * norm(llr_hat):
            break
        # Update result of last iteration
        llr_hat_old, x_hat_old = llr_hat, x_hat
    return x_hat.astype(int)
