import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, batch_first, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=batch_first, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(src)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, batch_first, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, batch_first=batch_first, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        if len(input.shape) < 3:
            input = input.unsqueeze(0)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len).to(self.device)
        hidden, cell = self.encoder(src)
        input = src[0, :]
        for t in range(0, batch_size):
            hidden_slice = hidden[:, t, :].unsqueeze(0)
            cell_slice = cell[:, t, :].unsqueeze(0)
            output, hidden_slice, cell_slice = self.decoder(input, hidden_slice, cell_slice)
            outputs[t, :] = output.squeeze(-1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else output.unsqueeze(0)
        return outputs
