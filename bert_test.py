from transformers import AutoTokenizer
from model import CustomBertClassifier
from data_preprocessing import bert_process

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

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
# print(tokenizer(['citation']))

ACL_TRAIN_PATH = './acl-arc/train.jsonl'
ACL_TEST_PATH = './acl-arc/test.jsonl'
ACL_DEV_PATH = './acl-arc/dev.jsonl'

train_data, test_data, dev_data = load_data(ACL_TRAIN_PATH), load_data(ACL_TEST_PATH), load_data(ACL_DEV_PATH)

# train_data, test_data, dev_data = train_data[:40], test_data, dev_data
bz = 290
# bertmodel_name = 'bert-large-uncased'
bertmodel_name = 'allenai/scibert_scivocab_uncased'
# bertmodel_name = 'bert-base-uncased'

if bertmodel_name == 'bert-base-uncased':
    bert_dim_size = 768
elif bertmodel_name == 'allenai/scibert_scivocab_uncased':
    bert_dim_size = 768
else:
    bert_dim_size = 1024


test = bert_process(test_data, batch_size=bz, pretrained_model_name=bertmodel_name)
test_loader = test.data_loader

network = CustomBertClassifier(hidden_dim= 100, bert_dim_size=bert_dim_size, num_of_output=6, model_name=bertmodel_name)

def evaluate_model(network, data, data_object):
    batch_size = 0
    f1s = []
    losses = []
    accus = []

    c = {str(i): 0 for i in range(6)}
    p = {str(i): 0 for i in range(6)}

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
        
        _, predicted = torch.max(output, dim=1)
        loss = accuracy_factor * loss_fn(output, y) + class_factor * torch.log((torch.subtract(y, predicted)!=0).sum())
        print("Accuracy Loss: ", accuracy_factor * loss_fn(output, y))
        print("Class Loss: ", class_factor * torch.log((torch.subtract(y, predicted) != 0).sum()))
        f1 = F1Score(num_classes=num_of_output, average='macro').to(device)
        f1_detailed = F1Score(num_classes=num_of_output, average='none').to(device)
        print("Specifically, ", f1_detailed(predicted, y))
        # self.output_types2idx = {'Background':3, 'Uses':1, 'CompareOrContrast':2, 'Extends':4, 'Motivation':0, 'Future':5}
        for x in y.cpu().detach().tolist():
            c[str(x)] += 1

        for pr in predicted.cpu().detach().tolist():
            p[str(pr)] += 1

        accuracy = Accuracy().to(device)
        f1 = f1(predicted, y)
        ac = accuracy(predicted, y)
        f1s.append(f1.cpu().detach().numpy())
        losses.append(loss.cpu().detach().numpy())
        accus.append(ac.cpu().detach().numpy())

    print('y_true: ', c)  
    print('y_pred: ',p)
    print('y_types: ',data_object.output_types2idx)  

    f1s = np.asarray(f1s)
    f1 = f1s.mean()
    accus = np.asarray(accus)
    losses = np.asarray(losses)
    accus = accus.mean()
    loss = losses.mean()
    print("Loss : %f, f1 : %f, accuracy: %f" % (loss, f1, accus))
    return f1

network.load_state_dict(torch.load("bestmodel.npy"))
# print("The best dev f1 is ", best_f1)
# network.eval()
print("The test f1 is")
evaluate_model(network, test_loader, test)