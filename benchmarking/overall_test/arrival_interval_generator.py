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

    arrival_times = np.random.exponential(scale=1/args.rps, size=args.num_reqs-1).tolist()

    fname = f'arrival_intervals-rps{args.rps}-reqs{args.num_reqs}-num{args.num}.json'
    with open(fname, "w") as f:
        json.dump(arrival_times, f, indent=4)
