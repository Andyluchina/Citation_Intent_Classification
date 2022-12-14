from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
print(tokenizer(['citation']))