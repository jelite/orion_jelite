import torch
import sys
from argparse import ArgumentParser


def cuda_sleep(mili):
    torch.cuda._sleep(int(mili * 1.54 * 10**6))


argparser = ArgumentParser()
argparser.add_argument("--long_pri", type=int)
argparser.add_argument("--short_pri", type=int)
argparser.add_argument("--short_iter", type=int)
args = argparser.parse_args()

high_stream = torch.cuda.Stream(priority=args.long_pri)
low_stream = torch.cuda.Stream(priority=args.short_pri)
low_iter = args.short_iter

high_stream.wait_stream(torch.cuda.current_stream())
low_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(high_stream):
    cuda_sleep(100)
with torch.cuda.stream(low_stream):
    for _ in range(100):
        cuda_sleep(1)
torch.cuda.current_stream().wait_stream(high_stream)
torch.cuda.current_stream().wait_stream(low_stream)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    high_stream.wait_stream(torch.cuda.current_stream())
    low_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(high_stream):
        cuda_sleep(100)
    with torch.cuda.stream(low_stream):
        for _ in range(low_iter):
            cuda_sleep(1)
    torch.cuda.current_stream().wait_stream(high_stream)
    torch.cuda.current_stream().wait_stream(low_stream)

torch.cuda.synchronize()

torch.cuda.profiler.cudart().cudaProfilerStart()
graph.replay()
torch.cuda.profiler.cudart().cudaProfilerStop()
print("done")
