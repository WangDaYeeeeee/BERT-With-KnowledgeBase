import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout_rate=0.):
        super(LSTMEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)

    def forward(self, seq_input, seq_lens):
        """
        Args:
            seq_input: Input of Bi-LSTM. Tensor[batch_size, max_seq_len, input_size]
            seq_lens: Sequence lengths of input. Tensor[batch_size]

        Return:
            seq_output: Tensor[batch_size, max_seq_len, num_dirs * hidden_size], (
                h_output: Tensor[num_layers * num_dirs, batch_size, hidden_size],
                c_output: Tensor[num_layers * num_dirs, batch_size, hidden_size]
            )
        """

        seq_input = self.dropout(seq_input)
        max_seq_len = seq_input.shape[1]

        desc_lens, desc_idx = seq_lens.sort(dim=0, descending=True)
        seq_input = seq_input[desc_idx]

        seq_input = pack_padded_sequence(seq_input, desc_lens, batch_first=True)
        seq_output, (h_output, c_output) = self.lstm(seq_input)
        seq_output, _ = pad_packed_sequence(seq_output, batch_first=True, total_length=max_seq_len)

        _, unsort_idx = desc_idx.sort(0, descending=False)
        seq_output = seq_output[unsort_idx]

        return seq_output, (h_output, c_output)


class LSTMDecoder(nn.Module):

    def __init__(self, args, input_size, hidden_size, num_layers=1, bidirectional=False, dropout_rate=0.):
        super(LSTMDecoder, self).__init__()

        self.args = args
        self.hidden_size = hidden_size
        self.num_dirs = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.)

    def forward(self, seq_input, seq_lens, h_input=None, c_input=None):
        """
        Args:
            seq_input: Input of LSTM. Tensor[batch_size, max_seq_len, input_size]
            seq_lens: Sequence lengths of input. Tensor[batch_size]
            h_input: h0 of LSTM. Tensor[num_layers * num_dirs, batch_size, hidden_size]
            c_input: c0 of LSTM. Tensor[num_layers * num_dirs, batch_size, hidden_size]

        Return:
            Tensor[batch_size, max_seq_len, hidden_size]
        """

        seq_input = self.dropout(seq_input)
        batch_size = seq_input.shape[0]
        max_seq_len = seq_input.shape[1]

        if h_input is None:
            h_input = torch.zeros(self.num_layers * self.num_dirs, batch_size, self.hidden_size).to(self.args.device)
        if c_input is None:
            c_input = torch.zeros(self.num_layers * self.num_dirs, batch_size, self.hidden_size).to(self.args.device)

        desc_lens, desc_idx = seq_lens.sort(dim=0, descending=True)
        seq_input = seq_input[desc_idx]

        seq_input = pack_padded_sequence(seq_input, desc_lens, batch_first=True)
        seq_output, _ = self.lstm(seq_input, (h_input, c_input))
        seq_output, _ = pad_packed_sequence(seq_output, batch_first=True, total_length=max_seq_len)

        _, unsort_idx = desc_idx.sort(0, descending=False)
        return seq_output[unsort_idx]
