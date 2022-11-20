# run main models
import torch
import json

# checking devices
device = None
if torch.cuda.is_available():
    print("Cuda is available, using CUDA")
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    print("MacOS acceleration is available, using MPS")
    device = torch.device('mps')
else:
    print("No acceleration device detected, using CPU")
    device = torch.device('cpu')

# print(device)
# load data
ACL_TRAIN_PATH = './ACL-ARC/training.jsonl'
data = []

f = open(ACL_TRAIN_PATH, "r")

for x in f:
  y = json.loads(x)
  data.append(y)
  print(y['cur_sent'])
  break
