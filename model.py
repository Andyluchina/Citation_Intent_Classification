# model code
from torch import nn, Tensor
import torch
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CustomTransformerClassifier(nn.Module):
  def __init__(self, enbedding_glove, MAXLEN = 200, output_class_num = 4, dim_feedforward=512, head=8, num_layers=3, dropout=0.2, d_model=256, freeze_emb = False):
    """
    args:
      enbedding_glove: pretrained glove embedding vector
    """
    ### Please do not change this function at all.
    super(CustomTransformerClassifier, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.pos_encoder = PositionalEncoding(d_model, dropout)
    # Embedding table over input vocabulary
    self.embedder = nn.Embedding.from_pretrained(enbedding_glove,freeze = freeze_emb)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,dropout=dropout,dim_feedforward=dim_feedforward, nhead=head)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.input_projection = nn.Linear(enbedding_glove.shape[1], d_model)
    self.output_linear = nn.Linear(MAXLEN*d_model,output_class_num)
    self.logsoftmax = nn.LogSoftmax(dim=1)
  def forward(self, sentence,src_mask = None,section_name = None):
    """
    args:
      sentence: [seq_len x batch_size] token indices
    returns:
      output_probs [batch_size x output_class_num]
    """
    seq_len, batch_size = sentence.size()
    # embedded_sentence: seq_len x batch_size x d_model
    embedded_sentence = self.input_projection(self.embedder(sentence))
    pos_emb = self.pos_encoder(embedded_sentence)
    # trans: seq_len x batch_size x d_model
    trans = self.transformer_encoder(pos_emb, src_mask)
    trans_permute = torch.permute(trans, (1, 0, 2))
    trans_flatten = torch.Flatten(trans_permute)
    trans_output = self.output_linear(trans_flatten)
    output_probs = self.logsoftmax(trans_output)
    return output_probs
