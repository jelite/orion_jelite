import glob
import pandas as pd
import numpy as np
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument("--dir", type=str)
argparser.add_argument("--tile", type=int)

args = argparser.parse_args()

data_list = []

for f in glob.glob(f"./be_infer/A_{args.dir}/*.txt"):

    infer = f.split("x")[1]
    infer = infer.split("_")[0]

    data = pd.read_csv(f, sep=",", header=None, names=["dur", "num"])

    tile = np.percentile(data, args.tile, method="nearest")
    tile = data["dur"].max()    
    print(f"{f},{tile}")