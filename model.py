from torch import nn, Tensor
import torch
import math
from transformers import BertModel




class CustomBertClassifier(nn.Module):
    def __init__(self, hidden_dim= 200, bert_dim_size=768, num_of_output=6, lstm_hidden = 200, model_name = "bert-base-uncased"):
        """

        """
        super(CustomBertClassifier, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(4*lstm_hidden, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_of_output)
        # self.bert_model = model
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.model = BertModel.from_pretrained(model_name)
        for name, param in self.model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
        self.lstm = nn.LSTM(input_size=bert_dim_size, hidden_size=lstm_hidden, num_layers=3, batch_first=False, dropout=0.2, bidirectional=True)
    def forward(self, sentences, citation_idxs, mask, device="mps"):
        """
        args:
            sentences: batch X seq_len
            citation_idxs: batch
            mask: batch X seq_len
        return:
            log_probs: batch X num_of_output
        """
        # bert.to(device)
        bert_output = self.model(input_ids=sentences, encoder_attention_mask=mask)
        # print(len(bert_output))
        # bert_output: batch X seq_len X bert_dim_size
        # print(bert_output[0].shape)
        # print(bert_output[1].shape)
        # first_tokens = bert_output[1]
        bert_output = bert_output[0]
        lstm_output = self.lstm(bert_output)
        lstm_output = lstm_output[0]
        # print(lstm_output.shape)
        # bert_output: batch X seq_len X 2*bert_dim_size
        # print(bert_output.shape)
        
        # print(citation_idxs)
        citation_tokens = lstm_output[torch.arange(bert_output.shape[0]), citation_idxs]
        # print(citation_tokens[0])
        first_tokens = lstm_output[:, 0]
        # print(bert_output[0, citation_idxs[0]])
        # print(first_tokens.shape)
        # print(citation_tokens.shape)
        # first_tokens batch X bert_dim_size
        concat_tokens = torch.concat((first_tokens, citation_tokens), dim=1)
        # concat_tokens = citation_tokens
        # concat_tokens batch X 2*bert_dim_size
        x1 = concat_tokens
        x2 = self.dropout(self.relu(self.linear1(x1)))
        # x3 = self.linear2(x2)
        x4 = self.linear3(x2)
        x5 = self.logsoftmax(x4)
        return x5
