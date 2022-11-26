import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import F1Score, Accuracy
from model import CustomBertClassifier
from data_preprocessing import bert_process
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter, OrderedDict
import torch.nn.functional as F


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




def load_data(path):

    data = []
    for x in open(path, "r"):
        data.append(json.loads(x))
    return data


ACL_TRAIN_PATH = './acl-arc/train.jsonl'
ACL_TEST_PATH = './acl-arc/test.jsonl'
ACL_DEV_PATH = './acl-arc/dev.jsonl'

train_data, test_data, dev_data = load_data(ACL_TRAIN_PATH), load_data(ACL_TEST_PATH), load_data(ACL_DEV_PATH)

SCICITE_TRAIN_PATH = './scicite/train.jsonl'
SCICITE_TEST_PATH = './scicite/test.jsonl'
SCICITE_DEV_PATH = './scicite/dev.jsonl'

train_data_sci, test_data_sci, dev_data_sci = load_data(SCICITE_TRAIN_PATH), load_data(SCICITE_TEST_PATH), load_data(SCICITE_DEV_PATH)

# train_data, test_data, dev_data = train_data[:40], test_data, dev_data
bz = 64
bertmodel_name = 'bert-base-uncased'
bert_dim_size = 768

train = bert_process(train_data, train_data_sci, batch_size=bz, pretrained_model_name=bertmodel_name)
# train = bert_process(train_data, batch_size=bz, pretrained_model_name=bertmodel_name)
train_loader = train.data_loader

dev = bert_process(dev_data, batch_size=bz, pretrained_model_name=bertmodel_name)
dev_loader = dev.data_loader

test = bert_process(test_data, batch_size=bz, pretrained_model_name=bertmodel_name)
test_loader = test.data_loader

num_of_output = 6


network = CustomBertClassifier(hidden_dim= 80, bert_dim_size=bert_dim_size, num_of_output=6)
# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.151702786,7.234782609,43.78947368,52.82539683,55.46666667]).to(device))
# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.006, 0.031, 0.043,0.32, 0.26,0.335]).to(device))

loss_fn = nn.NLLLoss()
# optimizer = torch.optim.Adam(network.parameters(), weight_decay = 1e-3, lr=0.01)
optimizer = torch.optim.Adam(network.parameters(), lr =0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 1, factor = 0.5, verbose = True)
n_epochs = 60

pytorch_total_params = sum(p.numel() for p in network.parameters())
# for parameter in network.parameters():
#     print(parameter)
print("all number of params ", pytorch_total_params)
pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
print("Trainable parameters " ,pytorch_total_params)
def evaluate_model(network, data, data_object):
    batch_size = 0
    f1s = []
    losses = []
    accus = []

    c=Counter()
    p = Counter()
    for batch in tqdm(data):
        x, y = batch
        network.eval()
        y = y.type(torch.LongTensor)
        y = y.to(device)
        sentences, citation_idxs, mask, token_id_types = x
        sentences, citation_idxs, mask, token_id_types = sentences.to(device), citation_idxs.to(device), mask.to(device),token_id_types.to(device)
        output = network(sentences, citation_idxs, mask, token_id_types, device=device)
        # loss = F.cross_entropy(output, y, weight=torch.tensor([1.0, 5.151702786,7.234782609,43.78947368,52.82539683,55.46666667]).to(device))
        # loss = F.nll_loss(output, y, weight=torch.tensor([1.0, 500.151702786,700.234782609,4300.78947368,5200.82539683,5500.46666667]).to(device))
        loss = loss_fn(output, y)
        _, predicted = torch.max(output, dim=1)
        f1 = F1Score(num_classes=num_of_output, average='macro').to(device)
        # print(predicted)
        # print(y)

        
        yy=y.cpu().detach().tolist()
        for x in yy:
            c.update([x])
        pre =predicted.cpu().detach().tolist()
        for pr in pre:
            p.update([pr])
        # print(y)
        # print(predicted)
        accuracy = Accuracy().to(device)
        f1 = f1(predicted, y)
        ac = accuracy(predicted, y)
        f1s.append(f1.cpu().detach().numpy())
        losses.append(loss.cpu().detach().numpy())
        accus.append(ac.cpu().detach().numpy())

    print('y: ', c)  
    print('pred: ',p)
    print('data_object.output_types2idx: ',data_object.output_types2idx)  

    f1s = np.asarray(f1s)
    f1 = f1s.mean()
    accus = np.asarray(accus)
    losses = np.asarray(losses)
    accus = accus.mean()
    loss = losses.mean()
    print("Loss : %f, f1 : %f, accuracy: %f" % (loss, f1, accus))
    return f1

best_f1 = -1
curr_f1 = -1
for epoch in range(n_epochs):
    print('Epoch', epoch)
    # train_loss = []
    for batch in tqdm(train_loader):
        x, y = batch
        network.train()
        assert network.training, 'make sure your network is in train mode with `.train()`'
        optimizer.zero_grad()
        network.to(device)
        y = y.type(torch.LongTensor)  
        y = y.to(device)
        sentences, citation_idxs, mask, token_id_types = x
        sentences, citation_idxs, mask, token_id_types = sentences.to(device), citation_idxs.to(device), mask.to(device),token_id_types.to(device)
        # print(sentences[0:2])
        # print(token_id_types[0:2])
        output = network(sentences, citation_idxs, mask, token_id_types, device=device)
        # print(output.shape)
        # print(y)
        # print(output)
        # loss = F.cross_entropy(output, y, weight=torch.tensor([1.0, 5.151702786,7.234782609,43.78947368,52.82539683,55.46666667]).to(device))
        loss = loss_fn(output, y)
        # loss = F.nll_loss(output, y, weight=torch.tensor([1.0, 500.151702786,700.234782609,4300.78947368,5200.82539683,5500.46666667]).to(device))
        # print(loss)
        loss.backward()
        optimizer.step()
    
    # print("The training loss is ", train_loss.mean())
    network.eval()
    print("train loss and f1")
    curr_f1 = evaluate_model(network, train_loader, train)
    print("dev loss and f1")
    curr_f1 = evaluate_model(network, dev_loader, dev)
    scheduler.step(curr_f1)
    if curr_f1 > best_f1:
        best_f1 = curr_f1
        torch.save(network.state_dict(), "bestmodel.npy")
    print("test loss and f1")
    evaluate_model(network, test_loader, test)

network.load_state_dict(torch.load("bestmodel.npy"))
print("The best dev f1 is ", best_f1)
network.eval()
print("The test f1 is")
evaluate_model(network, test_loader)