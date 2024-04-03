from argparse import ArgumentParser
from profiling.benchmarks.vision_models import vision

argparser = ArgumentParser()
argparser.add_argument("--model_name", type=str)
argparser.add_argument("--batch_size", type=int)
argparser.add_argument("--do_infer", action='store_true')
argparser.add_argument("--do_additional", action='store_true')
args = argparser.parse_args()

print(f"do infer_ {args.do_infer}")

vision(model_name=args.model_name, 
        batch_size=args.batch_size, 
        local_rank=0, 
        do_eval=args.do_infer, 
        is_additional=args.do_additional)

print("end")