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
   'Background',
   'Uses', 
   'Compare Or Contrast', 
   'Extends',
   'Motivation', 
   'Future'
]

encoded_labels = tokenizer(output_sematics, padding = 'max_length', max_length = 5, return_tensors='pt')

print(encoded_labels)

inputs = encoded_labels['input_ids']
mask = encoded_labels['attention_mask']

scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
scibert.to(device)

inputs = inputs.to(device)
mask =mask.to(device)

res = scibert(input_ids=inputs, attention_mask=mask)
bert_output = res[0]
first_tokens = bert_output[torch.arange(bert_output.shape[0]), 0]

print(first_tokens.shape)