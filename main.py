# run main models
import torch
import json
import numpy as np
import data_preprocessing as processing
from torchtext.vocab import GloVe

glove_vectors = GloVe(name="6B", dim=100)

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


# load data
ACL_TRAIN_PATH = './ACL-ARC/train.jsonl'
data = []

f = open(ACL_TRAIN_PATH, "r")

for x in f:
  y = json.loads(x)
  data.append(y)



# all the dinctionary keys in data:
# ['text', 'citing_paper_id', 'cited_paper_id', 'citing_paper_year', 'cited_paper_year', 'citing_paper_title', 'cited_paper_title', 'cited_author_ids', 
# 'citing_author_ids', 'extended_context', 'section_number', 'section_title', 'intent', 'cite_marker_offset', 'sents_before', 'sents_after', 
# 'cleaned_cite_text', 'citation_id', 'citation_excerpt_index', 'section_name']

kept_key = ['text','extended_context','intent','cleaned_cite_text','section_name']
filtered_data = processing.filter(data, kept_key)

keys = ['text','extended_context']
word_dicts = processing.make_word_dict(filtered_data, keys=keys)

text_dict, extended_context_dict = word_dicts


input_types, input_types2idx = processing.counts_to_vocab(text_dict, mfreq=1)


a = glove_vectors.get_vecs_by_tokens(input_types)
count = 0
for i, vec in enumerate(a):
    if torch.sum(vec)==0:
        count+=1
        print(input_types[i],text_dict[input_types[i]])
print(count)
print(len(input_types))
# print(a)

# print(input_types)