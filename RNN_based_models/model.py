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
        self.seq_len = args.seq_len
        self.embedding_dim = args.embed_dim
        self.embedding_num = args.embed_num

        self.embed = nn.Embedding(args.embed_num + 1, args.embed_dim)
        self.rnn = nn.LSTM(args.embed_dim, args.hidden_size, args.num_layers, batch_first=True)
        self.fc = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, input_x, seq):
        # Set initial states
        x = self.embed(input_x)  # dim: (batch_size, max_seq_len, embedding_size)
        packed_x = pack_padded_sequence(x, seq, batch_first=True)

        h0 = Variable(torch.rand(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        # Forward propagate RNN
        packed_h, (packed_h_t, packed_c_t) = self.rnn(packed_x, (h0,c0))
        # logit, _ = pad_packed_sequence(packed_h, batch_first=True)
        #
        # list_tensor = torch.cuda.LongTensor(len(seq)).zero_()
        # for i in range(len(seq)):
        #     list_tensor[i] = int(seq[i])
        # idx = (list_tensor - 1).view(-1, 1).expand(logit.size(0), logit.size(2)).unsqueeze(1)
        # decoded = logit.gather(1, idx).squeeze()

        #logit, (packed_h_t, packed_c_t) = self.rnn(x, (h0, c0))


        decoded = packed_h_t[-1]

        # Decode hidden state of last time step
        logit = self.fc(decoded)
        return F.log_softmax(logit)

