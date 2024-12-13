import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
model = 'resnet50_42.53'
df = pd.read_csv(f'/workspace/exps/be_infer/{model}.csv')

rps = []
dur_sum = 0
dur_len = 0
for dur in df['0']:
    dur_sum += dur
    dur_len += 1
    if(dur_sum > 60):
        rps.append(dur_len)
        dur_sum = 0
        dur_len = 0
print(f"len(rps): {len(rps)}")

# Convert rps list to DataFrame
rps_df = pd.DataFrame(rps)

# Save to CSV file, using same path/name pattern as input but with _rps suffix
output_path = '/workspace/exps/be_infer/rps_show.csv'
rps_df.to_csv(output_path, index=False)
print(f"Saved RPS data to: {output_path}")
