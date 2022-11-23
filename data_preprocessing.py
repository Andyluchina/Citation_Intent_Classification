"""process two sets of data"""

import torch
from collections import defaultdict, Counter, OrderedDict
import re
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
import numpy as np



# x = 'related work'
# splitted = re.split(' |-', x) # split by white space or dash
# out = [w[:4] if (len(w)==5 and w[:4].isnumeric()) else w for w in splitted] # modify some year-related string: 1997a -> 1997
# print(out)






class Dataset:

    def __init__(self, x, y, mask):
        # self.x = torch.tensor(x,dtype=torch.float32)
        # self.y = torch.tensor(y,dtype=torch.float32)
        # self.mask = torch.tensor(mask,dtype=torch.int32)
        self.x = x
        self.y = y
        self.mask = mask

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.y[idx]

    def __len__(self):
        return len(self.x)








class Process:


    def __init__(self, data:list[dict], input_key:list[str]=['section_name', 'cleaned_cite_text'], min_freq:int=1, max_len:int=250, 
                batch_size:int=1, shuffle:bool=True, glove_name:str="6B",glove_dim:int=100):


        self.data = data
        self.num_example = len(data)
        self.splitted_data = data

    # all the dinctionary keys in data:

    # ['text', 'citing_paper_id', 'cited_paper_id', 'citing_paper_year', 'cited_paper_year', 'citing_paper_title', 'cited_paper_title', 'cited_author_ids', 
    # 'citing_author_ids', 'extended_context', 'section_number', 'section_title', 'intent', 'cite_marker_offset', 'sents_before', 'sents_after', 
    # 'cleaned_cite_text', 'citation_id', 'citation_excerpt_index', 'section_name']

    # relevant keys: ['text','extended_context','intent','cleaned_cite_text','section_name']

    # 'section_name': {'introduction' , 'experiments', None: , 'conclusion', 'related work', 'method'}
    # 'intent': {'Background', 'Uses', 'CompareOrContrast', 'Extends', 'Motivation', 'Future'}


        self.max_len = max_len
        self.min_freq = min_freq

        self.input_key = input_key
        self.input_word_counter = Counter()
        self.input_types = np.array(["<UNK>"]) # initialize with UNK
        self.input_types2idx = {}
        self.indexed_input = []
        self.padded_input = torch.zeros(self.num_example, self.max_len) # (number of samples, max_len)
        self.mask = torch.ones_like(self.padded_input) # (# of samples, max_len)
        self.padding_idx = []  # (# of samples, various length)

        self.output_key = 'intent'
        self.output_word_counter = Counter()
        self.output_types = np.array([])
        self.output_types2idx = {}
        self.indexed_output = []

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = None
        self.data_loader = None

        self.glove_vectors = GloVe(name=glove_name, dim=glove_dim)
        self.pretrained_embedding = self.glove_vectors.get_vecs_by_tokens("<UNK>").view(1,-1) # initialize with UNK
        self.V = None

        self.make()




    def make_counters(self):

        for example in self.data:
            
            # for input
            for key in self.input_key:

                string = example[key] if example[key] != None else 'none'
                self.input_word_counter.update(self.split(string.lower())) 

            # for output
            self.output_word_counter.update([example[self.output_key]])





    def split(self, x:str):

        splitted = re.split(' |-', x) # split by white space or dash

        out = [w[:4] if (len(w)==5 and w[:4].isnumeric()) else w for w in splitted] # modify some year-related string: 1997a -> 1997

        return out




    def build_input_vocab(self):

        for key, count in self.input_word_counter.items():

            use_key = 'citation' if key == '@@citation' else key
            vec = self.glove_vectors.get_vecs_by_tokens(use_key)

            if any([ele != 0 for ele in vec]) and count >= self.min_freq:

                self.pretrained_embedding = torch.cat((self.pretrained_embedding, vec.view(1, -1)), 0)
                self.input_types = np.append(self.input_types, key)

        # get dict mapping word to idx
        self.input_types2idx = {w: i for i, w in enumerate(self.input_types)}


        # set the embedding for UNK to the avg of all other word embeddings
        self.V = self.pretrained_embedding.shape[0]
        UNK_index = self.input_types2idx["<UNK>"]

        self.pretrained_embedding[UNK_index, :] = torch.sum(self.pretrained_embedding[UNK_index+1:, :], 0).div(self.V-1)

        


    def build_output_vocab(self): 

        self.output_types = [wtype for (wtype, wcount) in self.output_word_counter.most_common()]

        self.output_types2idx = {wtype: i for i, wtype in enumerate(self.output_types)}




    def index_data(self):

        for example in self.data:
            
            # input
            x = ''
            for key in self.input_key:
                string = example[key] if example[key] != None else 'none'
                x += ' ' + string
            x = x[1:]
            indexed_x = [self.input_types2idx.get(w, self.input_types2idx["<UNK>"]) for w in self.split(x.lower())]
            self.indexed_input.append(torch.tensor(indexed_x))

            # output
            y = example[self.output_key]
            self.indexed_output.append(self.output_types2idx[y])




    def input_padding(self):

        for i, example in enumerate(self.indexed_input):

            example = example[:self.max_len]

            l = len(example)

            self.padded_input[i,:l] = example
            self.mask[i,l:] = 0
            self.padding_idx.append(torch.tensor([k for k in range(l, self.max_len)]))



    def prepare_dataset(self):

        self.dataset = Dataset(self.padded_input, self.indexed_output, self.mask)

        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)



    def make(self):

        self.make_counters()
        self.build_input_vocab()
        self.build_output_vocab()
        self.index_data()
        self.input_padding()
        self.prepare_dataset()


