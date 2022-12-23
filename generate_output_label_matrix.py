from transformers import AutoTokenizer
import torch
from transformers import AutoModel
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

output_sematics = [
   'Background: introduce related information about a subject',
   'Uses: introduce applications about a subject', 
   'Compare Or Contrast: compare the similarities and differences between the current subject and something else', 
   'Extends: introduces additions and extensions to the current subject',
   'Motivation: introduce reasons why certain subject is important', 
   'Future: introduce additional work that can be done in the future'
]

encoded_labels = tokenizer(output_sematics, padding = 'max_length', max_length =20)

print(encoded_labels)

inputs = encoded_labels['input_ids']
mask = encoded_labels['attention_mask']


print(tokenizer.decode(inputs[0]))

# scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
# scibert.to(device)

# inputs = inputs.to(device)
# mask =mask.to(device)

# res = scibert(input_ids=inputs, attention_mask=mask)
# bert_output = res[0]
# first_tokens = bert_output[torch.arange(bert_output.shape[0]), 0]

# print(first_tokens.shape)