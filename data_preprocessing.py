"""process two sets of data"""

import torch
from collections import defaultdict, Counter, OrderedDict
import re
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import BertTokenizer
import json
import matplotlib.pyplot as plt




class bert_process:

    def __init__(self, aclarc_data:list, scicite_data=None, confidence_level:float=1, cite2sentence_percent:float=0.15, max_len:int=300, 
    batch_size:int=1, shuffle:bool=True, pretrained_model_name:str='bert-base-uncased', padding:str='max_length'):
        
        self.data = aclarc_data
        self.scicite_data = scicite_data

        self.confidence_level = confidence_level
        self.cite2sentence_percent = cite2sentence_percent
        self.label_map = {'background':'Background','method':'Uses','result':'CompareOrContrast'}
        self.section_name_vocab = ['introduction','experiment','conclusion','related work','method','discussion','result','background']

        self.max_len = max_len
        self.padding = padding
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.citation_id = torch.tensor(11091) # id for citation
        self.sep_id = torch.tensor(102) # id for [SEP]
        self.cite_pos = [] #citation pisition

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexed_input = None
        self.indexed_output = None
        self.output_types2idx = {}
        self.mask = None
        self.token_type_ids = None

        if self.scicite_data:
            self.clean_add_scicite_data()
        self.index_input()
        self.index_output()
        self.make_data_loader()


    """
        # acl-arc data:

        # keys:
        # ['text', 'citing_paper_id', 'cited_paper_id', 'citing_paper_year', 'cited_paper_year', 'citing_paper_title', 'cited_paper_title', 'cited_author_ids', 'citing_author_ids', 'extended_context', 
        # 'section_number', 'section_title', 'intent', 'cite_marker_offset', 'sents_before', 'sents_after', 'cleaned_cite_text', 'citation_id', 'citation_excerpt_index', 'section_name']

        # relevant keys: ['text','extended_context','intent','cleaned_cite_text','section_name']

        # 'section_name': {'introduction' , 'experiments', None , 'conclusion', 'related work', 'method'}

        # 'intent': {'Background', 'Uses', 'CompareOrContrast', 'Extends', 'Motivation', 'Future'}




        # scicite data:

        # keys: ['source', 'citeEnd', 'sectionName', 'citeStart', 'string', 'label', 'label_confidence', 'citingPaperId', 'citedPaperId', 'isKeyCitation', 'id', 'unique_id', 'excerpt_index']

        # relevant keys: ['citeEnd', 'sectionName', 'citeStart', 'string', 'label', 'label_confidence']

        # 'sectionName': 
        # Counter({'Discussion': 1240, 'Introduction': 834, 'Methods': 800, '': 587, 'DISCUSSION': 483, 'Results': 359, '1. Introduction': 349, 'METHODS': 296, 'INTRODUCTION': 263, '4. Discussion': 172, 'RESULTS': 162, 
        # '1 Introduction': 160, 'Background': 101, 'Method': 87, 'Results and discussion': 74, '2. Methods': 60, '1. INTRODUCTION': 54, 'Results and Discussion': 41, 'Methodology': 37, 'Materials and methods': 35, 
        # 'RESULTS AND DISCUSSION': 33, '3. Discussion': 31, 'Materials and Methods': 30, '2 Methods': 25, nan: 19, '4 Experiments': 19, '3. Results': 17, 'Experimental design': 17, 'MATERIALS AND METHODS': 16, 
        # '4. DISCUSSION': 16, '5. Discussion': 16, '4 Discussion': 16, '5 Experiments': 16, 'Implementation': 15, 'Present Address:': 13, '2. Method': 13, '3 Experiments': 12, 'METHOD': 11, '6 Experiments': 11, 
        # '3. Methods': 11, '2 Related Work': 11, 'Experiments': 10, '4. Experiments': 10, '1 INTRODUCTION': 10, '3. Results and discussion': 9, 'Experimental Design': 7, '5. Experiments': 7, '3. Methodology': 7, 
        # '2. METHODS': 7, '1. Background': 7, '2. Results and Discussion': 7, '2. Related Work': 7, '2 Related work': 7, 'METHODOLOGY': 7, 'Discussion and conclusions': 7, 'Technical considerations': 6, '3 Methodology': 6,
        # '4 Implementation': 6, '3.2. Yield and characterisation of ethanol organosolv lignin (EOL)': 6, '3. Gaussian mixture and Gaussian mixture-of-experts': 6, '2 Method': 6, 
        # 'Effects of Discourse Stages on Adjustment of Reference Markers': 6, 'Identification of lysine propionylation in five bacteria': 6, '6. Mitochondrial Lesions': 5, '2.2 Behavior and Mission of Autonomous System': 5,
        # 'Structure and function of the placenta': 5, '5. A comparison with sound generation models': 5, 'Conclusions': 5, 'Role of Holistic Thinking': 5, '1.5 The Mystery of the Sushi Repeats: GABAB(1) Receptor Isoforms': 5, 
        # '3. EXPERIMENTS': 5, 'MBD': 5, '3.1. Depression': 5, 'Implications for Explaining Cooperation: Towards a Classification in Terms of Psychological Mechanisms': 4, 'Interpretation Bias in Social Anxiety': 4, '7 Experiments': 4,
        # '4 EXPERIMENTS': 4, '5.2. Implications for Formation of Crustal Plateau Structures': 4, '7 Numerical experiments': 4, 'Changes in CO2 and O3 Effects Through Time': 4, '3. A REAL OPTION TO INVEST UNDER CHOQUET-BROWNIAN AMBIGUITY': 4,
        # .........


        # 'label': Counter({'background': 4840, 'method': 2294, 'result': 1109})

    """
    

    def plot_sorted_label_confidence(self):
        plt.plot(sorted([exa.get('label_confidence', 0) for exa in self.scicite_data]))
        plt.show()


    def plot_sorted_citationLength_percentage(self):
        percent = []
        for exa in self.scicite_data:
            try:
                percent.append((exa['citeEnd']-exa['citeStart'])/len(exa['string']))
            except:
                percent.append(1.0)
        plt.plot(sorted(percent),'*')
        # plt.show()
        return sum(np.array(percent)>0.15)



    def standardized_section_name(self, scicite_section_name:str):

        corrected = re.sub(r'[^a-zA-Z\s]', '', scicite_section_name).lower() # only keep letters and whitespace
        
        for name in self.section_name_vocab:
            res = []
            if name in corrected:
                if len(corrected.split()) <= 5: 
                    return corrected
                res.append(name)

        if res: 
            return (' and '.join(res))
        return None



    def clean_add_scicite_data(self): # for scicite
        # conditions:
        # 1. has confidence level and >= 1
        # 2. label: 'result' to 'CompareOrContrast'; 'method' to 'uses' 
        # 3. citation len / sentence len <= 0.15; citation to @@citation 
        # 4. standardize section name 
        for exa in self.scicite_data:
            high_confi = exa.get('label_confidence', 0) >= self.confidence_level
            short_cite = isinstance(exa['citeEnd'],(float,int)) and isinstance(exa['citeStart'],(float,int)) and \
                        (exa['citeEnd']-exa['citeStart'])/len(exa['string']) <= self.cite2sentence_percent
            if high_confi and short_cite:
                exa['intent'] = self.label_map[exa['label']]
                print(exa['string'][int(exa['citeStart']):int(exa['citeEnd']+3)])
                start, end = int(exa['citeStart']), int(exa['citeEnd'])
                exa['cleaned_cite_text'] = exa['string'][:start] + "@@CITATION" + exa['string'][end:]
                exa['section_name'] = self.standardized_section_name(exa['sectionName'])
                self.data.append(exa)



    def index_input(self):
        
        raw_x = []
        for i, exa in enumerate(self.data):
            text, section_name = re.sub("@@CITATION", "@CITATION@", exa['cleaned_cite_text']), exa['section_name']
            raw_x.append('sentence : {} [SEP] section name : {}'.format(text, section_name))

        encoded_x = self.tokenizer(raw_x, padding=self.padding, max_length=self.max_len, return_tensors='pt') # dict
        self.indexed_input, self.mask, self.token_type_ids  = encoded_x['input_ids'], encoded_x['attention_mask'], encoded_x['token_type_ids']

        self.cite_pos = []
        for i, x_i in enumerate(self.indexed_input):
            for j,ele in enumerate(x_i):
                if ele == self.citation_id:
                    self.cite_pos.append(j)
                if ele == self.sep_id:
                    self.token_type_ids[i,j+1:] = 1
        self.cite_pos = torch.tensor(self.cite_pos)
        

    def index_output(self):

        c = 0
        self.indexed_output = np.array([])

        for exa in self.data:
            w = exa['intent']
            if self.output_types2idx.get(w, -1) == -1:
                self.output_types2idx[w] = c
                c += 1
            self.indexed_output = np.append(self.indexed_output, self.output_types2idx[w])


    def make_data_loader(self):

        dataset = Dataset(self.indexed_input, self.cite_pos, self.indexed_output, self.mask, self.token_type_ids)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)




def load_data(path):
    return [json.loads(x) for x in open(path, "r")]



# ACL_TRAIN_PATH = './ACL-ARC/train.jsonl'
# ACL_TEST_PATH = './ACL-ARC/test.jsonl'
# ACL_DEV_PATH = './ACL-ARC/dev.jsonl'

# train_data, test_data, dev_data = load_data(ACL_TRAIN_PATH), load_data(ACL_TEST_PATH), load_data(ACL_DEV_PATH)



# SCICITE_TRAIN_PATH = './scicite/train.jsonl'
# SCICITE_TEST_PATH = './scicite/test.jsonl'
# SCICITE_DEV_PATH = './scicite/dev.jsonl'

# train_data, test_data, dev_data = load_data(SCICITE_TRAIN_PATH), load_data(SCICITE_TEST_PATH), load_data(SCICITE_DEV_PATH)

# train: ~8k data points
# 6k has label confidence
# 4.3k label confidence >= 0.75
# 3.4k label confidence = 1
# whole='dasd dsa ds ly (Dymkowska et al. sfef fe '
# place='ly (Dymkowska et al.'
# a = whole[:12] + "@@CITATION" + whole[32:]
# print(whole[12:32])
# print(a)
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# model = BertModel.from_pretrained('bert-large-uncased')
# # text = train_data[0]['string']
# # # text='CITATION'
# # text='[SEP]'
# text = "Typical examples are Bulgarian ( @Citation@ ; Simov and Osenova , 2003 ) , [SEP] Chinese ( Chen et al. , 2003 ) , Danish ( Kromann , 2003 ) , and Swedish ( Nilsson et al. , 2005 ) . Second Sentence is here as well ."
# # # text = ["Typical examples are Bulgarian ( @Citation@ ; Simov and Osenova , 2003 ) , Chinese ( Chen et al. , 2003 ) , Danish ( Kromann , 2003 ) , and Swedish ( Nilsson et al. , 2005 ) . Second Sentence is here as well .",
# # # "Typical examples are Bulgarian ( @Citation@ ; Simov and Osenova , 2003 ) , Chinese ( Chen et al. , 2003 ) , Danish ( Kromann , 2003 ) , and Swedish ( Nilsson et al. , 2005 ) . Second Sentence is here as well ."]
# # print(text)
# encoded_input = tokenizer(text, padding='max_length', max_length=100)
# for key,val in encoded_input.items():
#     print(key,val)
# # print(tokenizer.decode(encoded_input['input_ids']))


# # Convert token to vocabulary indices
# # indexed_tokens = tokenizer.convert_tokens_to_ids(encoded_input)
# # print(indexed_tokens)


# aa = bert_process(None,train_data)
# # aa.plot_label_confidence_level()
# b =aa.plot_sorted_citationLength_percentage()
# print(b)





class Dataset:

    def __init__(self, x, citation_pos, y, mask, token_type_ids):
        self.x = x
        self.y = y
        self.citation_pos = citation_pos
        self.mask = mask
        self.token_type_ids = token_type_ids

    def __getitem__(self, idx):
        return (self.x[idx], self.citation_pos[idx], self.mask[idx], self.token_type_ids[idx]), self.y[idx]

    def __len__(self):
        return len(self.x)








"""

class glove_process:


    def __init__(self, data:list[dict], input_key:list[str]=['section_name', 'cleaned_cite_text'], min_freq:int=1, max_len:int=300, 
                batch_size:int=1, shuffle:bool=True, glove_name:str="6B", glove_dim:int=100):


        self.data = data # a list of dictionaries, one for one example/data point
        self.num_example = len(data) # number of data points in train/test/dev set

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

        # self.run()



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


"""