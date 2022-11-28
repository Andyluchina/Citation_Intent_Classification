from torch import nn, Tensor
import torch
import math
from transformers import BertModel




class CustomBertClassifier(nn.Module):
    def __init__(self, hidden_dim= 100, bert_dim_size=768, num_of_output=6, lstm_hidden = 100,proj_size=100, model_name = "bert-base-uncased"):
        """
        """
        super(CustomBertClassifier, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(2*lstm_hidden, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_of_output)
        self.linear_bert = nn.Linear(bert_dim_size, lstm_hidden)
        # self.bert_model = model
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.model = BertModel.from_pretrained(model_name)
        for name, param in self.model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False
        # self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, num_layers=4, batch_first=True, dropout=0.25)
        encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden, nhead=3, dim_feedforward=100, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    def forward(self, sentences, citation_idxs, mask, token_type_id=None, device="mps"):
        """
        args:
            sentences: batch X seq_len
            citation_idxs: batch
            mask: batch X seq_len
        return:
            log_probs: batch X num_of_output
        """
        # bert.to(device)
        bert_output = self.model(input_ids=sentences, attention_mask=mask, token_type_ids=token_type_id)
        # bert_output = self.model(input_ids=sentences, attention_mask=mask)
        # bert_output = self.model(input_ids=sentences)
        # print(len(bert_output))
        # bert_output: batch X seq_len X bert_dim_size
        # print(bert_output[0].shape)
        # print(bert_output[1].shape)
        # first_tokens = bert_output[1]
        bert_output = bert_output[0]
        # print(bert_output[:, -1].shape)
        # mask = mask.type(torch.Tensor).to(device)
        lstm_output = self.transformer_encoder(self.linear_bert(self.dropout(bert_output)))
        # lstm_output = lstm_output[0]
        # print(lstm_output.shape)
        # lstm_output: batch X seq_len X 2*bert_dim_size
        # print(bert_output.shape)
        
        citation_tokens = lstm_output[torch.arange(bert_output.shape[0]), citation_idxs]
        first_tokens = lstm_output[torch.arange(bert_output.shape[0]), 0]
        # print(first_tokens[0])
        # print(lstm_output[0,0])
        
        # first_tokens batch X bert_dim_size
        concat_tokens = torch.concat((first_tokens, citation_tokens), dim=1)
        # concat_tokens = torch.flatten(lstm_output,start_dim=1)
        # concat_tokens = citation_tokens
        # concat_tokens batch X 2*bert_dim_size
        x1 = concat_tokens
        x2 = self.dropout(self.relu(self.linear1(x1)))
        x3 = self.relu(self.linear3(x2))
        # x4 = self.linear3(x3)
        x5 = self.logsoftmax(x3)
        # print(torch.exp(x5))
        return x5