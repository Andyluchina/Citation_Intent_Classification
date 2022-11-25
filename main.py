# run main models

import torch
import json
import numpy as np
from data_preprocessing import Process
from collections import defaultdict, Counter, OrderedDict

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

# train_data, test_data, dev_data = load_data(ACL_TRAIN_PATH), load_data(ACL_TEST_PATH), load_data(ACL_DEV_PATH)






SCICITE_TRAIN_PATH = './SCICITE/train.jsonl'
SCICITE_TEST_PATH = './SCICITE/test.jsonl'
SCICITE_DEV_PATH = './SCICITE/dev.jsonl'

train_data, test_data, dev_data = load_data(SCICITE_TRAIN_PATH), load_data(SCICITE_TEST_PATH), load_data(SCICITE_DEV_PATH)

# train: ~8k data points
# 6k has label confidence
# 4.3k label confidence >= 0.75
# 3.4k label confidence = 1


print(len(train_data))
a=0
b=0
idx = []
for i, x in enumerate(train_data):
    if i==0:
        print(x['string'])
    try:
        print("================")
        print(x['string'][int(x['citeStart']):int(x['citeEnd'])], int(x['citeStart']), int(x['citeEnd']), len(x['string']))
        if (int(x['citeEnd']) - int(x['citeStart'])) / len(x['string']) >= 0.4:
            b+=1
    except:
        print("index nan")
        idx.append(i)
        a+=1
        print(x['citeStart'], x['citeEnd'])


    # try:
    #     if x['label_confidence'] >= 0.7 and x['label_confidence'] < 1:
    #         print(x['isKeyCitation'])
    #         continue
    #         # print(x)
    #         # break
    # except:
    #     # print(x)
    #     continue
    # break

print(a)
print(b)
# print(len(idx))


# c=Counter()
# for i,x in enumerate(train_data):
#     c.update([x['sectionName']])

# print(c)

# Counter({'Discussion': 1240, 'Introduction': 834, 'Methods': 800, '': 587, 'DISCUSSION': 483, 'Results': 359, '1. Introduction': 349, 'METHODS': 296, 'INTRODUCTION': 263, '4. Discussion': 172, 'RESULTS': 162, 
# '1 Introduction': 160, 'Background': 101, 'Method': 87, 'Results and discussion': 74, '2. Methods': 60, '1. INTRODUCTION': 54, 'Results and Discussion': 41, 'Methodology': 37, 'Materials and methods': 35, 
# 'RESULTS AND DISCUSSION': 33, '3. Discussion': 31, 'Materials and Methods': 30, '2 Methods': 25, nan: 19, '4 Experiments': 19, '3. Results': 17, 'Experimental design': 17, 'MATERIALS AND METHODS': 16, 
# '4. DISCUSSION': 16, '5. Discussion': 16, '4 Discussion': 16, '5 Experiments': 16, 'Implementation': 15, 'Present Address:': 13, '2. Method': 13, '3 Experiments': 12, 'METHOD': 11, '6 Experiments': 11, 
# '3. Methods': 11, '2 Related Work': 11, 'Experiments': 10, '4. Experiments': 10, '1 INTRODUCTION': 10, '3. Results and discussion': 9, 'Experimental Design': 7, '5. Experiments': 7, '3. Methodology': 7, 
# '2. METHODS': 7, '1. Background': 7, '2. Results and Discussion': 7, '2. Related Work': 7, '2 Related work': 7, 'METHODOLOGY': 7, 'Discussion and conclusions': 7, 'Technical considerations': 6, '3 Methodology': 6,
# '4 Implementation': 6, '3.2. Yield and characterisation of ethanol organosolv lignin (EOL)': 6, '3. Gaussian mixture and Gaussian mixture-of-experts': 6, '2 Method': 6, 
# 'Effects of Discourse Stages on Adjustment of Reference Markers': 6, 'Identification of lysine propionylation in five bacteria': 6, '6. Mitochondrial Lesions': 5, '2.2 Behavior and Mission of Autonomous System': 5,
# 'Structure and function of the placenta': 5, '5. A comparison with sound generation models': 5, 'Conclusions': 5, 'Role of Holistic Thinking': 5, '1.5 The Mystery of the Sushi Repeats: GABAB(1) Receptor Isoforms': 5, 
# '3. EXPERIMENTS': 5, 'MBD': 5, '3.1. Depression': 5, 'Implications for Explaining Cooperation: Towards a Classification in Terms of Psychological Mechanisms': 4, 'Interpretation Bias in Social Anxiety': 4, '7 Experiments': 4,
# '4 EXPERIMENTS': 4, '5.2. Implications for Formation of Crustal Plateau Structures': 4, '7 Numerical experiments': 4, 'Changes in CO2 and O3 Effects Through Time': 4, '3. A REAL OPTION TO INVEST UNDER CHOQUET-BROWNIAN AMBIGUITY': 4,
# ...........






# d=Counter()
# for i,x in enumerate(train_data):
#     d.update([x['label']])

# print(d)

# Counter({'background': 4840, 'method': 2294, 'result': 1109})








# create a class to process data and hold attributes
# train = Process(train_data, shuffle=False, batch_size=2)



# for j, (x_train, mask, y_train) in enumerate(train.data_loader):
#     # print(j,(x_train,mask, y_train))
#     print(np.take(train.input_types, x_train))
#     break






# dev = Process(dev_data)

# test = Process(test_data)

