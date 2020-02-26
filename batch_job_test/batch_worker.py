import sys
sys.path.append('recursive_RM')
import os
import argparse as ap
import pickle
from recursive_RM import *
from itertools import product, combinations
import xarray as xr

parser = ap.ArgumentParser('Batch worker for RM decode simulation')
parser.add_argument('worker_id', help='Integer id identifying the run')

# parse arguments
args = parser.parse_args()
worker_id = int(args.worker_id)
r, m = 3, 7
# Random seed to generate samples
seed = worker_id
# Maximum number of samples to simulate
maxCw = 100
# Minimum number of‰ word error to terminate
minCwErr = 10
# Set of Eb_No to simulate, in unit dB
dB_range = [2, 2.5, 3, 3.5]
# List size for the list decoder, 0 if do not use list decoder
list_size = 4

# save to file
save_filename = 'results/r={}_m={}_{}.pickle'.format(r, m, worker_id)

# Generate generator matrix and parity-check matrix
Gsys, perm, inv_perm = systemize(RM(r, m))
H = RM(m - r - 1, m)

# Setup decoder keyword arguments
default_kwargs = {'weight': {}, 'damp': 1, 'llr_clip': 30}
algor_details = {
    'RPA': ({**default_kwargs, 'base_dec': 'soft'}, RPA),
    'CPA': ({**default_kwargs, 'base_dec': 'soft'}, CPA),
    'RXA': (default_kwargs, RXA),
    'CXA': (default_kwargs, CXA)
}

# Initialize dataset for simulation results
metrics = ['info_bit_error', 'bit_error', 'word_error', 'ML_lower_bound']
metrics = metrics + [m + '_list' for m in metrics]
res_shape = (len(algor_details), len(dB_range), maxCw, len(metrics))

result_ds = xr.Dataset({
    'r': r, 'm': m, 'seed': seed,
    'result': xr.DataArray(
        np.full(res_shape, np.nan),
        dims=('algorithm', 'SNR_dB', 'sample_id', 'metric'),
        coords={
            'algorithm': list(algor_details.keys()),
            'SNR_dB': dB_range, 'metric': metrics
        }
    ),
    'kwargs': xr.DataArray(
        [kwargs for kwargs, decoder in algor_details.values()],
        dims=('algorithm'),
        coords={'algorithm': list(algor_details.keys())}
    )
})


# Generate list decoder
def list_decoder_factory(decoder, decoder_kwargs, list_size):
    # Reed's Algorithm
    def reeds_algorithm(f, r, m):
        ''' Reed's algorithm to decode Reed-Muller codes
        Arguments:
            f -- input binary vector
            r -- degree of Reed-Muller code
            m -- number of variables of Reed-Muller code
        Returns:
            P -- decoded codeword
        '''
        X = np.array(list(product([0, 1], repeat=m)), dtype=int).T[::-1]
        F, P, t = f.copy(), np.zeros_like(f), r
        while t >= 0:
            for S in combinations(list(range(m)), t):
                S_bar, temp = np.array([i for i in range(m) if i not in S]), 0
                for b in product([0, 1], repeat=m - t):
                    ind = (X[S_bar] == np.array(b)[:, None]).all(axis=0)
                    temp += F[ind].sum() % 2
                if temp >= 2 ** (m - t - 1):
                    mono = np.bitwise_and.reduce(X[np.array(S)], axis=0) if t > 0 else 1
                    P, F = P ^ mono, F ^ mono
            t -= 1
        return P

    def decode_func(llr):
        def D_func(llr):
            return decoder(llr, r, m, **kwargs)
        # original decoder
        c_hat = D_func(llr)
        c_hat_reeds = reeds_algorithm(c_hat, r, m)
        # list decoder
        L_tilde, L_try = llr, llr.copy()
        # find index of weakest bits
        L_max, weak_idx = 2 * abs(llr).max(), abs(llr).argsort()[:list_size]
        # use original decoder result as benchmark
        c_hat_list = c_hat_reeds
        score = np.dot((-1) ** c_hat_list, L_tilde)
        # enumerate all possible guess of weakest bits
        for u in product([-L_max, L_max], repeat=list_size):
            L_try[weak_idx] = np.array(u)
            c_hat_list_u = D_func(L_try)
            c_hat_list_u_reeds = reeds_algorithm(c_hat_list_u, r, m)
            score_u = np.dot((-1) ** c_hat_list_u_reeds, L_tilde)
            if score_u > score:
                c_hat_list, score = c_hat_list_u_reeds, score_u
        return c_hat, c_hat_reeds, c_hat_list
    return lambda llr: [out[perm] for out in decode_func(llr[inv_perm])]


rng = np.random.RandomState(seed)
for i, EbNo_dB in enumerate(dB_range):
    for t in range(maxCw):
        _, c, _, llr = AWGN_output(Gsys, EbNo_dB, rng)
        for algor, (kwargs, decoder) in algor_details.items():
            res = result_ds.result.loc[algor, EbNo_dB, t]
            # if there is any metric missing
            if np.isnan(res).any():
                # generate list decoder
                list_decoder = list_decoder_factory(decoder, kwargs, list_size)
                # decode the LLR sequence
                c_hat, c_hat_reeds, c_hat_list = list_decoder(llr)
                # fill the result array
                diff, diff_list = abs(c - c_hat), abs(c - c_hat_list)
                res.loc['info_bit_error'] = diff[:Gsys.shape[0]].sum()
                res.loc['bit_error'] = diff.sum()
                res.loc['word_error'] = diff.any()
                res.loc['ML_lower_bound'] = np.dot((-1) ** c, llr) < np.dot((-1) ** c_hat_reeds, llr)
                res.loc['info_bit_error_list'] = diff_list[:Gsys.shape[0]].sum()
                res.loc['bit_error_list'] = diff_list.sum()
                res.loc['word_error_list'] = diff_list.any()
                res.loc['ML_lower_bound_list'] = np.dot((-1) ** c, llr) < np.dot((-1) ** c_hat_list, llr)
        res = result_ds.result.sel(SNR_dB=EbNo_dB).sum(dim='sample_id')
        print(t)
        print(res.sel(metric=['word_error', 'word_error_list']))
        print(res.sel(metric=['bit_error', 'bit_error_list']))
        with open(save_filename, 'wb') as f:
            pickle.dump(result_ds, f)
