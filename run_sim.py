import time
import datetime
import argparse as ap
import json
from recursive_RM import *
from itertools import product, combinations

# Setup parser
parser = ap.ArgumentParser('Simulate RPA/CPA/RXA/CXA Decoders for Reed-Muller Codes')
parser.add_argument('--verbose', '-v', help='Display text output', action="store_true")
parser.add_argument('-r', type=int, default=2, help='Order of RM code')
parser.add_argument('-m', type=int, default=5, help='Size of RM code')
parser.add_argument('-l', '--list',  type=int,  dest='list_size_log2', default=0, help='Log2 of list size')
parser.add_argument('-d','--dec', dest='name', default='CPA', help='Name of decoding algorithm')
parser.add_argument('-w','--weight', nargs='+', dest='weight', type=float, default=None, help='Scale factor for variable node update')
parser.add_argument('--hard', help='Use hard-decision component decoding', action="store_true")
parser.add_argument('-c','--clip', dest='clip', type=float, default=30.0, help='LLR clipping value')
parser.add_argument('-n','--nblock', nargs='+', type=int, dest='maxCw', default=[100], help='Maximum number of codewords to simulate')
parser.add_argument('-t','--iter', type=int, dest='tmax', default=15, help='Maximum number of iterations')
parser.add_argument('--maxerr', nargs='+', type=int, dest='minCwErr', default=[10], help='Maximum number of errors to simulate')
parser.add_argument('-e','--ebno', nargs='+', dest='dB_range', type=float, default=[2.0], help='List of EbN0s to simulate')
parser.add_argument('-o','--out', dest='save_filename', default=None, help='Filename for JSON results')
parser.add_argument('-s','--seed', dest='seed', type=int, default=None, help='Seed for RNG')

# parse arguments
args = parser.parse_args()

# Random seed to generate samples
if (args.seed is None):
    vars(args).update({"seed":int(time.time()/1000.0)})

# Check and fix sizes
if (len(args.maxCw) != len(args.dB_range)):
    maxCw = [args.maxCw[0], ]*len(args.dB_range)
    vars(args).update({"maxCw":maxCw})
if (len(args.minCwErr) != len(args.dB_range)):
    minCwErr = [args.minCwErr[0], ]*len(args.dB_range)
    vars(args).update({"minCwErr":minCwErr})

# # save file name
# if (args.save_filename is None):
#     timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
#     vars(args).update({"save_filename":'R{}M{}L{}_{}_{}.json'.format(args.r, args.m, args.list_size_log2, args.name, timestamp)})

# Instantiate arguments as local variables for simplicity
locals().update(vars(args))

# Compute generator and parity-check matrices
Gsys, perm, inv_perm = systemize(RM(r, m))
H = RM(m - r - 1, m)

# Handle weight
wdict = {}
if (name[0]=="C"):
    if (weight is not None):
        wdict = {(r,m): weight[0]}
if (name[0]=="R"):
    if (weight is not None):
        if (len(weight) == r-1):
            for i in range(r-1):
                wdict = {(r-i,m-i): weight[i]}
        else:
            print("Weight vector incorrect length")

# Setup decoder keyword arguments
style = 'soft'
if (hard):
    style = 'hard'
default_kwargs = {'weight': wdict, 'damp': 1, 'llr_clip': clip, 't_max': tmax}
algor_details = {
    'RPA': ({**default_kwargs, 'base_dec': style}, RPA),
    'CPA': ({**default_kwargs, 'base_dec': style}, CPA),
    'RXA': (default_kwargs, RXA),
    'CXA': (default_kwargs, CXA)
}

# Get decoder
kwargs, decoder = algor_details[name] 

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

# Setup metrics
metrics = ["info_bit_error","bit_error","word_error","bit_error_reeds",
           "word_error_reeds","ML_lower_bound","info_bit_error_list",
           "bit_error_list","word_error_list","ML_lower_bound_list"]

# Outer simulation loop
rng = np.random.RandomState(seed)
results = []
for i, EbNo_dB in enumerate(dB_range):
    # generate list decoder
    list_decoder = list_decoder_factory(decoder, kwargs, list_size_log2)

    # setup results
    results.append({x: [] for x in metrics})

    while (len(results[i]["word_error_list"]) < maxCw[i] and np.sum(results[i]["word_error_list"]) < minCwErr[i]):
        # print if verbose
        if (verbose and len(results[i]["word_error_list"])%10 == 0):
            print("n="+str(len(results[i]["word_error_list"]))+" blkerr="+str(np.sum(results[i]["word_error_list"])))
        # generate the LLR sequence
        _, c, _, llr = AWGN_output(Gsys, EbNo_dB, rng)
        # decode the LLR sequence
        c_hat, c_hat_reeds, c_hat_list = list_decoder(llr)
        # fill the result array
        diff, diff_reeds, diff_list = abs(c - c_hat), abs(c - c_hat_reeds), abs(c - c_hat_list)
        results[i]["info_bit_error"].append( diff[:Gsys.shape[0]].sum() )
        results[i]["bit_error"].append( diff.sum() )
        results[i]["word_error"].append( diff.any() )
        results[i]["bit_error_reeds"].append( diff_reeds.sum() )
        results[i]["word_error_reeds"].append( diff_reeds.any() )
        results[i]["ML_lower_bound"].append( np.dot((-1) ** c, llr) < np.dot((-1) ** c_hat_reeds, llr) )
        results[i]["info_bit_error_list"].append( diff_list[:Gsys.shape[0]].sum() )
        results[i]["bit_error_list"].append( diff_list.sum() )
        results[i]["word_error_list"].append( diff_list.any() )
        results[i]["ML_lower_bound_list"].append( np.dot((-1) ** c, llr) < np.dot((-1) ** c_hat_list, llr) )

    # Summarize if verbose
    if (verbose):
        print({k:np.sum(v) for (k,v) in results[i].items()})
        
# Class to allow JSON to handle numpy types
class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return int(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(npEncoder, self).default(obj)

# Save with append or print to stdout
if (save_filename is not None):
    with open(save_filename, 'at') as f:
        f.write(json.dumps(vars(args),cls=npEncoder)+'\n'+json.dumps(results,cls=npEncoder)+'\n')
else:
    print("EbN0: "+str(dB_range))
    nblock = [len(results[i][metrics[1]]) for (i,_) in enumerate(dB_range)]
    print("nBlock: "+str(nblock))
    for metric in metrics:
        temp = []
        for i, EbNo_dB in enumerate(dB_range):
            val = np.sum(results[i][metric])
            count = nblock[i]
            temp.append(val)
        print(metric+": "+str(temp))
