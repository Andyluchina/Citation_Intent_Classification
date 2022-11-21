# run main models
import torch
import json
from torchtext.vocab import GloVe, vocab
import torchtext
# glove_vectors = GloVe("6B", dim = 100)

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

# print(device)
# load data
ACL_TRAIN_PATH = './acl-arc/train.jsonl'
data = []

SCICITE_TRAIN_PATH = './scicite/train.jsonl'

# f = open(ACL_TRAIN_PATH, "r")
#
#
# # dict_keys(['text',
# # 'citing_paper_id', 'cited_paper_id', 'citing_paper_year', 'cited_paper_year', 'citing_paper_title',
# # 'cited_paper_title', 'cited_author_ids', 'citing_author_ids', 'extended_context', 'section_number', 'section_title',
# # 'intent', 'cite_marker_offset', 'sents_before', 'sents_after', 'cleaned_cite_text', 'citation_id', 'citation_excerpt_index',
# # 'section_name'])
# for x in f:
#   y = json.loads(x)
#   data.append(y)
#   print(y['text'])
#   print(y['citing_paper_id'])
#   print(y['section_name'])
#   print(y['intent'])
#   print(y['citing_paper_title'])
#   print(y['extended_context'])
#   print(y['section_number']) #maybe.....
#   print(y['cite_marker_offset']) #where the interested citation is
#   print(y['cleaned_cite_text'])
#   # print(y['text'][134:148])
#   break
# # print(len(data))
#
f = open(SCICITE_TRAIN_PATH, "r")


# dict_keys(['text',
# 'citing_paper_id', 'cited_paper_id', 'citing_paper_year', 'cited_paper_year', 'citing_paper_title',
# 'cited_paper_title', 'cited_author_ids', 'citing_author_ids', 'extended_context', 'section_number', 'section_title',
# 'intent', 'cite_marker_offset', 'sents_before', 'sents_after', 'cleaned_cite_text', 'citation_id', 'citation_excerpt_index',
# 'section_name'])
for x in f:
  y = json.loads(x)
  data.append(y)
  print(y['string'])
  s = y['citeStart']
  e = y['citeEnd']
  print(y['string'][s:e])
  # print(y['text'][134:148])
  # break
