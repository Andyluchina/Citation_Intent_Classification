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

class CustomTransformerEncoder(nn.Module):
  def __init__(self, enbedding_glove, dim_feedforward=512, head=8, num_layers=3, dropout=0.2, d_model=256):
    """
    args:
      hidden_dim: hidden state size of LSTM
      dropout: this is applied to the output of the LSTM
    """
    ### Please do not change this function at all.
    super(CustomTransformerEncoder, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.pos_encoder = PositionalEncoding(d_model, dropout)
    # Embedding table over input vocabulary
    self.embedder = nn.Embedding.from_pretrained(enbedding_glove,freeze = False)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,dropout=dropout,dim_feedforward=dim_feedforward, nhead=head)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

  def forward(self, sentence):
    """
    args:
      sentence: [seq_len x batch_size] token indices

    returns:

    """
    seq_len, batch_size = sentence.size()

    # embedded_sentence: seq_len x batch_size x word_vector_dim
    embedded_sentence = self.embedder(sentence)
    pos_emb = self.pos_encoder(embedded_sentence)
    # lstm_out: seq_len x batch_size x 2 * hidden_dim
    # h_n, c_n: 2 x batch_size x hidden_dim
    trans = self.transformer_encoder(pos_emb)

    return trans
