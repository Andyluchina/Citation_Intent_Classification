from pytorch_transformers import BertTokenizer, BertModel
import torch

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

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', padding='right', max_length=20)
model = BertModel.from_pretrained('bert-large-uncased')
text = "Typical examples are Bulgarian ( @Citation@ ; Simov and Osenova , 2003 ) , Chinese ( Chen et al. , 2003 ) , Danish ( Kromann , 2003 ) , and Swedish ( Nilsson et al. , 2005 ) . Second Sentence is here as well ."
encoded_input = tokenizer(text)
# Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(encoded_input)
print(len(encoded_input))
# print(tokenizer(text))
# print(len(indexed_tokens))
# tokens_tensor = torch.tensor([indexed_tokens])
# print(tokens_tensor.shape)
# tokens_tensor = tokens_tensor.to(device)
# model.to(device)
# output = model(tokens_tensor)
# print(output[0].shape)