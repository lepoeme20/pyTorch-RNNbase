import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.seq_len = args.seq_len
        self.embedding_dim = args.embed_dim
        self.embedding_num = args.embed_num
        self.rnn_cell = args.which_rnn.lower()
        self.bidirectional = args.bidirectional

        self.embed = nn.Embedding(self.embedding_num + 1, self.embedding_dim)
        if self.rnn_cell == 'vanilla':
            self.rnn = nn.RNN(self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True,
                              bidirectional=self.bidirectional)
        elif self.rnn_cell == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=self.bidirectional)
        elif self.rnn_cell == 'gru':
            self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True,
                              bidirectional=self.bidirectional)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)

        if self.bidirectional:
            self.num_layers *= 2

    def forward(self, input_x, seq):
        # Set initial states
        x = self.embed(input_x)  # dim: (batch_size, max_seq_len, embedding_size)
        packed_x = pack_padded_sequence(x, seq, batch_first=True)

        h0 = Variable(torch.rand(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        # Forward propagate RNN
        if self.rnn_cell == 'lstm':
            packed_h, (packed_h_t, packed_c_t) = self.rnn(packed_x, (h0, c0))
        else:
            packed_h, packed_h_t = self.rnn(packed_x, h0)
        decoded = packed_h_t[-1]

        # Decode hidden state of last time step
        logit = self.fc(decoded)
        return F.log_softmax(logit)

