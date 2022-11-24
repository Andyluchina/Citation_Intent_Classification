from torch import nn, Tensor
import torch
import math
from transformers import BertModel
model = BertModel.from_pretrained("bert-large-uncased")

class CustomBertClassifier(nn.Module):
    def __init__(self, hidden_dim, bert_dim_size=756, num_of_output=6):
        """

        """
        super(CustomBertClassifier, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(2*bert_dim_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_of_output)
        self.bert_model = model
        
    def forward(self, sentences, citation_idxs, mask):
        """
        args:
            sentences: batch X seq_len
            citation_idxs: batch
            mask: batch X seq_len
        return:
            log_probs: batch X num_of_output
        """
        bert_output = self.bert_model(input_ids=sentences, encoder_attention_mask=mask)
        # bert_output: batch X seq_len X bert_dim_size

