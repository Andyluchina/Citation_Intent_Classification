"""process two sets of data"""

import torch
from collections import defaultdict, Counter, OrderedDict
import re
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe




# def filter(data: list[dict], keys: list):
#     return [{key: value for (key, value) in sample.items() if key in keys} for sample in data]


# def counts_to_vocab(counter:dict, mfreq:int=1) -> tuple[list, dict]: 

#     types = [wtype for (wtype, wcount) in counter.most_common() if wcount >= mfreq]
#     types2idx = {wordtype: i for i, wordtype in enumerate(types)}
#     return types, types2idx



# def make_input_counter(data: list[dict], key: str):

#     word_counter = Counter()
#     for example in data:
#         word_counter.update(split(example[key].lower()))
#     return word_counter



# def split(x:str):

#     splitted = re.split(' |-', x) # split by space and dash
#     res = [item[:4] if (len(item)==5 and item[:4].isnumeric()) else item for item in splitted] # year_modified 1997a -> 1997
#     return res




# def make_output_counter(data: list[dict]):

#     word_counter = Counter()
#     for example in data:
#         word_counter.update([example['intent']])
#     return word_counter



# def apply_glove(vocab_counts:dict, vectors:GloVe, mfreq:int=1):

#     # initialize with UNK
#     pretrained_embedding = vectors.get_vecs_by_tokens("<UNK>").view(1,-1)
#     types = ["<UNK>"]

#     # add words
#     for key, count in vocab_counts.items():
#         vec = vectors.get_vecs_by_tokens(key)
#         if any([ele != 0 for ele in vec]) and count >= mfreq:
#             pretrained_embedding = torch.cat((pretrained_embedding, vec.view(1,-1)),0)
#             types.append(key)

#     # embedding for unknow is avg of all other word embedding
#     pretrained_embedding[0,:] = torch.sum(pretrained_embedding[1:,:],0).div(pretrained_embedding.shape[0]-1)

#     types2idx = {word: i for i, word in enumerate(types)}
#     return pretrained_embedding, types, types2idx




# def index_data(data:list[dict], input_types2idx:dict, output_types2idx:dict) ->tuple[list[torch.tensor], list[int]]:

#     input, output = [], []
#     for example in data:
#         x, y = example['text'], example['intent']
#         input.append(torch.tensor([input_types2idx.get(word, input_types2idx["<UNK>"]) for word in split(x.lower())]))
#         output.append(output_types2idx[y])
#     return input, output



# def pad_input(train_x_indexed: list[torch.tensor]):

#     padded_input = pad_sequence(train_x_indexed).T # make batch first, then max length
#     max_len = padded_input.shape[1]
#     mask, padding_idx = [], []
#     for exa in train_x_indexed:
#         mask.append(torch.ones_like(exa))
#         padding_idx.append(torch.tensor([i for i in range(len(exa), max_len)]))

#     mask = pad_sequence(mask).T
#     return padded_input, mask, padding_idx # (batch, max_len), (batch, max_len), (batch, various_len)




def make_data_loader(data:list, batch_size:int=1):

    # filtered_data = filter(data, kept_key)


    # input_count = make_input_counter(filtered_data, 'text')

    # glove_vectors = GloVe(name="6B", dim=100)
    # glove_pretrained, input_types, input_types2idx = apply_glove(input_count, glove_vectors, mfreq=1) # apply glove to input x


    # output_count = make_output_counter(filtered_data)

    # output_types, output_types2idx = counts_to_vocab(output_count)



    # x_indexed, y_indexed = index_data(filtered_data, input_types2idx, output_types2idx)



    padded_x, mask, padding_idx = pad_input(x_indexed)


    # train_data = dataset(padded_x, y_indexed, mask)

    # data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # for j,(x_train,y_train) in enumerate(x):
    #     print(j,(x_train,y_train))
    #     break

    # return data_loader





class Process:

    def __init__(self, data:list[dict], input_key:str='text', min_freq:int=1, 
    max_len:int=250, batch_size:int=1, shuffle:bool=True, glove_name:str="6B",glove_dim:int=100):

        self.data = data

    # all the dinctionary keys in data:

    # ['text', 'citing_paper_id', 'cited_paper_id', 'citing_paper_year', 'cited_paper_year', 'citing_paper_title', 'cited_paper_title', 'cited_author_ids', 
    # 'citing_author_ids', 'extended_context', 'section_number', 'section_title', 'intent', 'cite_marker_offset', 'sents_before', 'sents_after', 
    # 'cleaned_cite_text', 'citation_id', 'citation_excerpt_index', 'section_name']

    # relevant keys ['text','extended_context','intent','cleaned_cite_text','section_name']

        self.input_key = input_key
        self.input_word_counter = Counter()
        self.input_types = []
        self.input_types2idx = {}
        self.indexed_input = []

        self.output_key = 'intent'
        self.output_word_counter = Counter()
        self.output_types = []
        self.output_types2idx = {}
        self.indexed_output = []

        self.max_len = max_len
        self.min_freq = min_freq

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.glove_vectors = GloVe(name=glove_name, dim=glove_dim)
        self.pretrained_embedding = None



    # def filter(self):
    #     return [{key: value for (key, value) in sample.items() if key in self.kept_key} for sample in self.data]


    def make_input_counter(self):
        for example in self.data:
            self.input_word_counter.update(self.split(example[self.input_key].lower()))

    
    def make_output_counter(self):
        for example in self.data:
            self.output_word_counter.update([example[self.output_key]])



    def split(self, x:str):
        splitted = re.split(' |-', x) # split by space and dash
        out = [w[:4] if (len(w)==5 and w[:4].isnumeric()) else w for w in splitted] # year_modified 1997a -> 1997
        return out



    def apply_glove(self):

        # initialize with UNK
        self.pretrained_embedding = self.glove_vectors.get_vecs_by_tokens("<UNK>").view(1,-1)
        self.input_types = ["<UNK>"]

        # add words
        for key, count in self.input_word_counter.items():
            vec = self.glove_vectors.get_vecs_by_tokens(key)
            if any([ele != 0 for ele in vec]) and count >= self.min_freq:
                pretrained_embedding = torch.cat((pretrained_embedding, vec.view(1,-1)),0)
                self.input_types.append(key)

        # embedding for unknow is avg of all other word embedding
        pretrained_embedding[0,:] = torch.sum(pretrained_embedding[1:,:],0).div(pretrained_embedding.shape[0]-1)

        self.input_types2idx = {word: i for i, word in enumerate(self.input_types)}



    def counts_to_vocab(self): 
        self.output_types = [wtype for (wtype, wcount) in self.output_word_counter.most_common() if wcount >= self.min_freq]
        self.output_types2idx = {wtype: i for i, wtype in enumerate(self.output_types)}



    def index_data(self):

        for example in self.data:
            x, y = example[self.input_key], example[self.output_key]
            indexed_x = [self.input_types2idx.get(w, self.input_types2idx["<UNK>"]) for w in self.split(x.lower())]
            self.indexed_input.append(torch.tensor(indexed_x))
            self.indexed_output.append(self.output_types2idx[y])


    def input_padding(self): # need change!!!!! self.max_len=250
        # padded_input = pad_sequence(self.indexed_input).T # make batch first, then max length
        # max_len = padded_input.shape[1]
        # mask, padding_idx = [], []
        # for example in self.indexed_input:
        #     mask.append(torch.ones_like(example))
        #     padding_idx.append(torch.tensor([i for i in range(len(example), max_len)]))

        # mask = pad_sequence(mask).T
        # return padded_input, mask, padding_idx # (batch, max_len), (batch, max_len), (batch, various_len)

        self.padded_input





    def make(self): # call above class functions to generate things, like a script

        self.make_input_counter()
        self.apply_glove()
        self.make_output_counter()
        self.counts_to_vocab()
        self.index_data()
        self.input_padding()

        self.dataset = Dataset(padded_x, y_indexed, mask)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)





class Dataset:

    def __init__(self, x, y, mask):
        # self.x = torch.tensor(x,dtype=torch.float32)
        # self.y = torch.tensor(y,dtype=torch.float32)
        # self.mask = torch.tensor(mask,dtype=torch.int32)
        self.x = x
        self.y = y
        self.mask = mask

    def __getitem__(self, idx):
        return (self.x[idx], self.mask[idx]), self.y[idx]

    def __len__(self):
        return len(self.x)