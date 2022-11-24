from transformers import BertTokenizer, BertModel
import json
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained("bert-large-uncased")
text = "Typical examples are Bulgarian ( @citation@ , 2005 ; Simov and Osenova , 2003 ) , Chinese ( Chen et al. , 2003 ) , Danish ( Kromann , 2003 ) , and Swedish ( Nilsson et al. , 2005 ) ."
encoded_input = tokenizer(text, padding="max_length", max_length=100)
# encoded_input = tokenizer(text)
print(encoded_input)
# print(tokenizer.tokenize(text))
# print(encoded_input)
print(tokenizer.decode(encoded_input["input_ids"]))
# output = model(**encoded_input)

ACL_TRAIN_PATH = './acl-arc/dev.jsonl'
data = []

f = open(ACL_TRAIN_PATH, "r")


# dict_keys(['text',
# 'citing_paper_id', 'cited_paper_id', 'citing_paper_year', 'cited_paper_year', 'citing_paper_title',
# 'cited_paper_title', 'cited_author_ids', 'citing_author_ids', 'extended_context', 'section_number', 'section_title',
# 'intent', 'cite_marker_offset', 'sents_before', 'sents_after', 'cleaned_cite_text', 'citation_id', 'citation_excerpt_index',
# 'section_name'])
max_len = 0
for x in f:
    y = json.loads(x)
    data.append(y)
    l = len(tokenizer(y['text'])['input_ids'])
    if l > max_len:
        max_len = l

print(max_len)
    #   print(y['text'])
    #   print(y['citing_paper_id'])
    #   print(y['section_name'])
    #   print(y['intent'])
    #   print(y['citing_paper_title'])
    #   print(y['extended_context'])
    #   print(y['section_number']) #maybe.....
    #   print(y['cite_marker_offset']) #where the interested citation is
    #   print(y['cleaned_cite_text'])
    # if y['section_name'] == None:
    #     print(y['text'])
    #     print(y['extended_context'])
    #     break