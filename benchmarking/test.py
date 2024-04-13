import json

path = './overall_test/arrival_times-rps85.05-reqs2000-num0.json'
with open(path, "r") as json_file:
        json_data = json.load(json_file)
print(json_data[0])