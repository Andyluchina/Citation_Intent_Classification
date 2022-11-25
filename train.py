import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import F1Score
from model import CustomBertClassifier
from data_preprocessing import bert_process
import json
import numpy as np


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

# train_data, test_data, dev_data = train_data[:40], test_data, dev_data
bz = 20
bertmodel_name = 'bert-base-uncased'
bert_dim_size = 768

train = bert_process(train_data, batch_size=bz, pretrained_model_name=bertmodel_name)
train_loader = train.data_loader

dev = bert_process(dev_data, batch_size=bz, pretrained_model_name=bertmodel_name)
dev_loader = dev.data_loader

test = bert_process(test_data, batch_size=bz, pretrained_model_name=bertmodel_name)
test_loader = test.data_loader

num_of_output = 6


network = CustomBertClassifier(hidden_dim= 100, bert_dim_size=bert_dim_size, num_of_output=6)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 2, factor = 0.3, verbose = True)
n_epochs = 60

def evaluate_model(network, data):
    batch_size = 0
    f1s = []
    losses = []
    for batch in tqdm(data):
        x, y = batch
        network.eval()
        y = y.to(device)
        sentences, citation_idxs, mask = x
        sentences, citation_idxs, mask = sentences.to(device), citation_idxs.to(device), mask.to(device)
        output = network(sentences, citation_idxs, mask, device=device)
        loss = loss_fn(output, y)
        _, predicted = torch.max(output, dim=1)
        f1 = F1Score(num_classes=num_of_output).to(device)
        f1 = f1(predicted, y)
        f1s.append(f1.cpu().detach().numpy())
        losses.append(loss.cpu().detach().numpy())
    f1s = np.asarray(f1s)
    f1 = f1s.mean()
    losses = np.asarray(losses)
    loss = losses.mean()
    print("Loss : %f, f1 : %f" % (loss, f1))
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
        y = y.to(device)
        sentences, citation_idxs, mask = x
        sentences, citation_idxs, mask = sentences.to(device), citation_idxs.to(device), mask.to(device)
        # print(x)
        output = network(sentences, citation_idxs, mask, device=device)
        # print(output.shape)
        loss = loss_fn(output, y)
        # print(loss)
        loss.backward()
        optimizer.step()
    
    # print("The training loss is ", train_loss.mean())
    network.eval()
    print("train loss and f1")
    curr_f1 = evaluate_model(network, train_loader)
    print("dev loss and f1")
    curr_f1 = evaluate_model(network, dev_loader)
    scheduler.step(curr_f1)
    if curr_f1 > best_f1:
        best_f1 = curr_f1
        torch.save(network.state_dict(), "bestmodel.npy")
    print("test loss and f1")
    evaluate_model(network, test_loader)

network.load_state_dict(torch.load("bestmodel.npy"))
print("The best dev f1 is ", best_f1)
network.eval()
print("The test f1 is")
evaluate_model(network, test_loader)