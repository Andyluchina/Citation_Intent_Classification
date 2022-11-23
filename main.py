# run main models

import torch
import json
import numpy as np
from data_preprocessing import Process



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

print(device)




def load_data(path):

    data = []
    for x in open(path, "r"):
        data.append(json.loads(x))
    return data


ACL_TRAIN_PATH = './ACL-ARC/train.jsonl'
ACL_TEST_PATH = './ACL-ARC/test.jsonl'
ACL_DEV_PATH = './ACL-ARC/dev.jsonl'

train_data, test_data, dev_data = load_data(ACL_TRAIN_PATH), load_data(ACL_TEST_PATH), load_data(ACL_DEV_PATH)






# create a class to process data and hold attributes
train = Process(train_data, shuffle=False, batch_size=2)



for j, (x_train, mask, y_train) in enumerate(train.data_loader):
    # print(j,(x_train,mask, y_train))
    print(np.take(train.input_types, x_train))
    break





# dev = Process(dev_data)

# test = Process(test_data)

