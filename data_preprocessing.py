"""process two sets of data"""

import torch
from collections import defaultdict, Counter, OrderedDict
import re
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
import numpy as np




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


        self.data = data # a list of dictionaries, one for one example/data point
        self.num_example = len(data) # number of data points in train/test/dev set

    # all the dinctionary keys in data:

    # ['text', 'citing_paper_id', 'cited_paper_id', 'citing_paper_year', 'cited_paper_year', 'citing_paper_title', 'cited_paper_title', 'cited_author_ids', 
    # 'citing_author_ids', 'extended_context', 'section_number', 'section_title', 'intent', 'cite_marker_offset', 'sents_before', 'sents_after', 
    # 'cleaned_cite_text', 'citation_id', 'citation_excerpt_index', 'section_name']

    # relevant keys: ['text','extended_context','intent','cleaned_cite_text','section_name']

    # 'section_name': {'introduction' , 'experiments', None: , 'conclusion', 'related work', 'method'}
    # 'intent': {'Background', 'Uses', 'CompareOrContrast', 'Extends', 'Motivation', 'Future'}


        self.max_len = max_len # for batching, the max length of input sequence of one sample
        self.min_freq = min_freq # min frequency for a word, set to unk if < min frequency

        self.input_key = input_key # which dict keys of raw data to use to construct input sequence
        self.input_word_counter = Counter()
        self.input_types = np.array(["<UNK>"]) # vocab, initialize with UNK
        self.input_types2idx = {}  # vocab words to index
        self.indexed_input = []     # integer input sequences that are indexed by input vocab
        self.padded_input = torch.zeros(self.num_example, self.max_len) # (number of samples, max_len)
        self.mask = torch.ones_like(self.padded_input) # (number of samples, max_len)
        self.padding_idx = []  # (number of samples, various length)

        self.output_key = 'intent'
        self.output_word_counter = Counter()
        self.output_types = np.array([])
        self.output_types2idx = {}
        self.indexed_output = []

        self.batch_size = batch_size # for data loader
        self.shuffle = shuffle
        self.dataset = None
        self.data_loader = None

        self.glove_vectors = GloVe(name=glove_name, dim=glove_dim)
        self.pretrained_embedding = self.glove_vectors.get_vecs_by_tokens("<UNK>").view(1,-1) # initialize with UNK
        self.V = None

        self.run() # call to run like a script





    # make word counters for input sequences and output words in data
    def make_counters(self):

        for example in self.data:
            
            # for input
            for key in self.input_key: 

                string = example[key] if example[key] != None else 'none' # for section name, example[key] could be None, so change it to the word 'none'
                self.input_word_counter.update(self.split(string.lower())) 

            # for output
            self.output_word_counter.update([example[self.output_key]])





    def split(self, x:str):

        splitted = re.split(' |-', x) # split by white space or dash; '|' can hold multiple delimiters

        out = [w[:4] if (len(w)==5 and w[:4].isnumeric()) else w for w in splitted] # modify some year-related strings: 1997a -> 1997

        return out




    def build_input_vocab(self):

        for word, count in self.input_word_counter.items():

            this_word = 'citation' if word == '@@citation' else word # search CITATION marker as 'citation' in Glove 
            vec = self.glove_vectors.get_vecs_by_tokens(this_word)

            if any([ele != 0 for ele in vec]) and count >= self.min_freq: # check if this word is in Glove and check its frequency

                self.pretrained_embedding = torch.cat((self.pretrained_embedding, vec.view(1, -1)), 0)
                self.input_types = np.append(self.input_types, word)

        # get dict, mapping word to idx
        self.input_types2idx = {w: i for i, w in enumerate(self.input_types)}

        # set the embedding for UNK to the avg of all other word embeddings
        self.V = self.pretrained_embedding.shape[0] # vocab size
        UNK_index = self.input_types2idx["<UNK>"]

        self.pretrained_embedding[UNK_index, :] = torch.sum(self.pretrained_embedding[UNK_index+1:, :], 0).div(self.V-1) # UNK was put as the first word

        


    def build_output_vocab(self): 

        self.output_types = [word for (word, count) in self.output_word_counter.most_common()]

        self.output_types2idx = {word: i for i, word in enumerate(self.output_types)}




    def index_data(self):

        for example in self.data:
            
            # input
            x = ''
            for key in self.input_key: # put 'citation_excerpt_index' and 'section_name' together as input
                string = example[key] if example[key] != None else 'none'
                x += ' ' + string
            x = x[1:] # remove first space
            indexed_x = [self.input_types2idx.get(w, self.input_types2idx["<UNK>"]) for w in self.split(x.lower())] # set to UNK if not in vocab that was built
            self.indexed_input.append(torch.tensor(indexed_x))

            # output
            y = example[self.output_key]
            self.indexed_output.append(self.output_types2idx[y])




    def input_padding(self):

        for i, example in enumerate(self.indexed_input):

            example = example[:self.max_len] # truncate if excced length; does nothing otherwise

            l = len(example)

            self.padded_input[i,:l] = example
            self.mask[i,l:] = 0     # broadcast to all 0
            self.padding_idx.append(torch.tensor([k for k in range(l, self.max_len)]))



    def prepare_dataset(self):

        self.dataset = Dataset(self.padded_input, self.indexed_output, self.mask)

        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)



    def run(self):

        self.make_counters()
        self.build_input_vocab()
        self.build_output_vocab()
        self.index_data()
        self.input_padding()
        self.prepare_dataset()


