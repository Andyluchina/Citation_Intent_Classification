from torch import nn, Tensor
import torch
import math
from transformers import BertModel


model = BertModel.from_pretrained("bert-base-uncased")

class CustomBertClassifier(nn.Module):
    def __init__(self, hidden_dim= 200, bert_dim_size=768, num_of_output=6):
        """

        """
        super(CustomBertClassifier, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(2*bert_dim_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_of_output)
        # self.bert_model = model
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, sentences, citation_idxs, mask, bert=model, device="mps"):
        """
        args:
            sentences: batch X seq_len
            citation_idxs: batch
            mask: batch X seq_len
        return:
            log_probs: batch X num_of_output
        """
        bert.to(device)
        bert_output = bert(input_ids=sentences, encoder_attention_mask=mask)
        # print(len(bert_output))
        # bert_output: batch X seq_len X bert_dim_size
        bert_output = bert_output[0]
        # print(bert_output.shape)
        first_tokens = bert_output[:, 0]
        # print(citation_idxs)
        citation_tokens = bert_output[torch.arange(bert_output.shape[0]), citation_idxs]
        # print(first_tokens.shape)
        # print(citation_tokens.shape)
        # first_tokens batch X bert_dim_size
        concat_tokens = torch.concat((first_tokens, citation_tokens), dim=1)
        # concat_tokens batch X 2*bert_dim_size
        x1 = self.dropout(concat_tokens)
        x2 = self.dropout(self.relu(self.linear1(x1)))
        x3 = self.linear2(x2)
        x4 = self.linear3(x3)
        x5 = self.logsoftmax(x4)
        return x5
