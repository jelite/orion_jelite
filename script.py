import sys
from profiling.benchmarks.vision_models import vision

model_name = sys.argv[1]
batch_size = int(sys.argv[2])
do_eval = bool(sys.argv[3])
profile = sys.argv[4] #ncu, nsys
vision(model_name, batch_size, 0, do_eval, profile)
print("end")