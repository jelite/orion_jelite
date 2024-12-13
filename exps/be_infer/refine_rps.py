import json


# Save merged traces
with open(f'traces-rps41.52-reqs20000-num0.json', 'r') as f:
    traces = json.load(f)
trace_sum = 0
for idx, trace in enumerate(traces):
    trace_sum += trace 
    if idx in [918, 918+788, 918+788+771, 918+788+771+795, 918+788+771+795+925, 918+788+771+795+925+1092]:
        print((idx+1)/trace_sum)
    # json.dump(traces, f, indent=4)

# import pdb; pdb.set_trace()
# requests = []
# for idx, trace in enumerate(traces):
#     if idx > 7900:
#         break
#     requests.append(trace)
#     if idx % (int)(7900/5) == 0 and idx != 0:
#         print(len(requests)/sum(requests))
#         requests = []