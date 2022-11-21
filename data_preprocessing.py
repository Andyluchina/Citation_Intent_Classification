# process two sets of data

from collections import defaultdict, Counter, OrderedDict



def filter(data: list[dict], keys: list):
    return [{key: value for (key, value) in sample.items() if key in keys} for sample in data]




def counts_to_vocab(counter:dict, mfreq:int) -> tuple[list, dict]: 

    UNK_SYMBOL = "<UNK>"
    # SOS_SYMBOL = "<SOS>"
    # EOS_SYMBOL = "<EOS>"
    types = [UNK_SYMBOL] # SOS_SYMBOL, EOS_SYMBOL
    types += [wtype for (wtype, wcount) in counter.most_common() if wcount >= mfreq]
    type2idx = {wordtype: i for i, wordtype in enumerate(types)}
    return types, type2idx


def make_word_dict(data: list[dict], keys: list):

    res = []
    for key in keys:
        word_counter = Counter()
        for example in data:
            word_counter.update(example[key].split())
        res.append(word_counter)
    return res

