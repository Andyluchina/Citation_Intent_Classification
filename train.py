import torch
import torch.nn as nn
import tqdm
from torchmetrics import F1Score
from model import CustomBertClassifier
from data_preprocessing import bert_process
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




def load_data(path):

    data = []
    for x in open(path, "r"):
        data.append(json.loads(x))
    return data


ACL_TRAIN_PATH = './ACL-ARC/train.jsonl'
ACL_TEST_PATH = './ACL-ARC/test.jsonl'
ACL_DEV_PATH = './ACL-ARC/dev.jsonl'

train_data, test_data, dev_data = load_data(ACL_TRAIN_PATH), load_data(ACL_TEST_PATH), load_data(ACL_DEV_PATH)




train = bert_process(train_data, batch_size=64)
train_loader = train.data_loader

dev = bert_process(dev_data, batch_size=len(dev_data))
dev_loader = dev.data_loader

test = bert_process(test_data, batch_size=len(test_data))
test_loader = test.data_loader

num_of_output = 6


network = CustomBertClassifier(hidden_dim= 200, bert_dim_size=768, num_of_output=6)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(network.parameters())
n_epochs = 60

def evaluate_model(network, data):
    x, y = data
    network.eval()
    y = y.to(device)
    sentences, citation_idxs, mask = x
    sentences, citation_idxs, mask = sentences.to(device), citation_idxs.to(device), mask.to(device)
    output = network(sentences, citation_idxs, mask, device=device)
    loss = loss_fn(output, y)
    _, predicted = torch.max(output, dim=1)
    f1 = F1Score(num_classes=num_of_output)
    f1 = f1(predicted, y)
    print("Loss : %f, f1 : %f \n" % (loss, f1))
    return f1

best_f1 = -1
curr_f1 = -1
for epoch in range(n_epochs):
    print('Epoch', epoch)
    for batch in tqdm.tqdm_notebook(train_loader, leave=False):
        x, y = batch
        network.train()
        assert network.training, 'make sure your network is in train mode with `.train()`'
        optimizer.zero_grad()
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
    network.eval()
    print("dev loss and f1")
    curr_f1 = evaluate_model(network, dev)
    if curr_f1 > best_f1:
        best_f1 = curr_f1
        torch.save(network.state_dict(), "bestmodel.npy")
    print("test loss and f1")
    evaluate_model(network, test)

network.load_state_dict(torch.load("bestmodel.npy"))
print("The best dev f1 is ", best_f1)
network.eval()
print("The test f1 is")
evaluate_model(network, test)