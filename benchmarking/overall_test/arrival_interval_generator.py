import numpy as np


import json
from argparse import ArgumentParser
import time

np.random.seed(time.time_ns() % 1000000000)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--rps", type=float, default=19.71)
    argparser.add_argument("--num_reqs", type=int, default=2000)
    argparser.add_argument("--num", type=int, default=0)
    args = argparser.parse_args()

    rps = args.rps

    interval = 1/rps
    arrival_times = np.random.uniform(interval - interval*0.25, interval + interval*0.25, args.num_reqs-1).tolist()
    fname = f'arrival_intervals_B025rt-rps{rps}-reqs{args.num_reqs}-num{args.num}.json'

    # arrival_times = np.random.exponential(scale=1/args.rps, size=10000).tolist()
    # fname = f'arrival_intervals_Bbe-rps{args.rps}-reqs{10000}-num{args.num}.json'

    with open(fname, "w") as f:
        json.dump(arrival_times, f, indent=4)
