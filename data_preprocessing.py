"""process two sets of data"""

import torch
from collections import defaultdict, Counter, OrderedDict
import re
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import BertTokenizer, BertModel






# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# # model = BertModel.from_pretrained('bert-large-uncased')

# # text='CITATION'
# # text = "Typical examples are Bulgarian ( @Citation@ ; Simov and Osenova , 2003 ) , Chinese ( Chen et al. , 2003 ) , Danish ( Kromann , 2003 ) , and Swedish ( Nilsson et al. , 2005 ) . Second Sentence is here as well ."
# text = ["Typical examples are Bulgarian ( @Citation@ ; Simov and Osenova , 2003 ) , Chinese ( Chen et al. , 2003 ) , Danish ( Kromann , 2003 ) , and Swedish ( Nilsson et al. , 2005 ) . Second Sentence is here as well .",
# "Typical examples are Bulgarian ( @Citation@ ; Simov and Osenova , 2003 ) , Chinese ( Chen et al. , 2003 ) , Danish ( Kromann , 2003 ) , and Swedish ( Nilsson et al. , 2005 ) . Second Sentence is here as well ."]

# encoded_input = tokenizer(text, padding='max_length', max_length=100)
# for key,val in encoded_input.items():
#     print(key,val)

# # Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(encoded_input)
# print(indexed_tokens)




class bert_process:

    def __init__(self, data, max_len:int=300, batch_size:int=1, shuffle:bool=True, pretrained_model_name:str='bert-large-uncased', padding:str='max_length'):
        
        self.data = data
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.padding = padding
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.citation_id = 11091 # id for citation
        self.cite_pos = [] #citation pisition
        self.indexed_input = None
        self.indexed_output = None
        self.output_types2idx = {}
        self.mask = None

        self.index_input()
        self.index_output()
        self.make_data_loader()


    def index_input(self):
        
        raw_input = []
        for i, example in enumerate(self.data):
            text, section_name = re.sub("@@CITATION", "@CITATION@", example['cleaned_cite_text']), example['section_name']
            raw_input.append('section name : {} [SEP] sentence: {}'.format(section_name, text))

        # input
        encoded_input = self.tokenizer(raw_input, padding=self.padding, max_length=self.max_len, return_tensors='pt') # dict
        self.indexed_input = encoded_input['input_ids']
        self.mask = encoded_input['attention_mask']
        
        self.cite_pos = []
        for i, row in enumerate(encoded_input['input_ids']):
            for j,ele in enumerate(row):
                if ele == torch.tensor(self.citation_id):
                    self.cite_pos.append(j)
        self.cite_pos = torch.tensor(self.cite_pos)




    def index_output(self):
        c = 0
        self.indexed_output = []

        for i, example in enumerate(self.data):
            w = example['intent']
            if self.output_types2idx.get(w, -1) == -1:
                self.output_types2idx[w] = c
                c += 1
            self.indexed_output.append(self.output_types2idx[w])

        self.indexed_output = np.array(self.indexed_output)



    def make_data_loader(self):
        dataset = Dataset(self.indexed_input, self.cite_pos, self.indexed_output, self.mask)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)






class Dataset:

    def __init__(self, x, pos, y, mask):
        # self.x = torch.tensor(x,dtype=torch.float32)
        # self.y = torch.tensor(y,dtype=torch.float32)
        # self.mask = torch.tensor(mask,dtype=torch.int32)
        self.x = x
        self.y = y
        self.pos = pos
        self.mask = mask

    def __getitem__(self, idx):
        return (self.x[idx], self.pos[idx], self.mask[idx]), self.y[idx]

    def __len__(self):
        return len(self.x)













        # self.run()


# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', padding='right', max_length=20)
# model = BertModel.from_pretrained('bert-large-uncased')

# text = "Typical examples are Bulgarian ( @Citation@ ; Simov and Osenova , 2003 ) , Chinese ( Chen et al. , 2003 ) , Danish ( Kromann , 2003 ) , and Swedish ( Nilsson et al. , 2005 ) . Second Sentence is here as well ."
# encoded_input = tokenizer(text)
# # Convert token to vocabulary indices
# # indexed_tokens = tokenizer.convert_tokens_to_ids(encoded_input)


    # def bertProcess(self):
    #     pass


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


