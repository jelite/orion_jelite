import torch
# Warm-up CUDA.
torch.empty(1, device="cuda")

# From test/test_cuda.py in PyTorch.
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
torch.cuda._sleep(1000000)
end.record()
end.synchronize()
cycles_per_ms = 1000000 / start.elapsed_time(end)
print(cycles_per_ms)
def cuda_sleep(seconds):
    torch.cuda._sleep(int(seconds * 1234592.2450473267 * 1000))
