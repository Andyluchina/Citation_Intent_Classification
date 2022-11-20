# model code
class TransformerEncoder(nn.Module):
  def __init__(self, enbedding_glove, dim_feedforward=512, head=8, num_layers=6, dropout=0.2, d_model=256):
    """
    args:
      hidden_dim: hidden state size of LSTM
      dropout: this is applied to the output of the LSTM
    """
    ### Please do not change this function at all.
    super(RNNEncoder, self).__init__()
    self.dropout = nn.Dropout(dropout)
    # Embedding table over input vocabulary
    self.embedder = nn.Embedding.from_pretrained(enbedding_glove,freeze = False)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,dropout=dropout,dim_feedforward=dim_feedforward, nhead=head)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

  def forward(self, sentence):
    """
    args:
      sentence: [seq_len x batch_size] token indices (in our case batch_size = 1)

    returns:
      lstm_out: [seq_len x batch_size x 2 * hidden_dim] (2 since bidirectional)
      h_n, c_n: [1 x batch_size x hidden_dim] final states to be passed to the decoder
    """
    seq_len, batch_size = sentence.size()

    # embedded_sentence: seq_len x batch_size x word_vector_dim
    embedded_sentence = self.embedder(sentence)

    # lstm_out: seq_len x batch_size x 2 * hidden_dim
    # h_n, c_n: 2 x batch_size x hidden_dim
    lstm_out, (h_n, c_n) = self.lstm(embedded_sentence)

    # Take average of states across forward and reverse directions.
    h_n = h_n.mean(0, keepdim=True)
    c_n = c_n.mean(0, keepdim=True)

    return self.dropout(lstm_out), (h_n, c_n)
