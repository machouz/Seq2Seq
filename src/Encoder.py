from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from src.utils import *


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(input_size, hidden_size // 2,)
        self.rnn = nn.GRU(hidden_size // 2, hidden_size, num_layers=num_layers, dropout=0.5,
                          bidirectional=bidirectional, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_size, device=device) # (num_layers * num_directions, batch, hidden_size)

    def forward_sequence(self, input):
        input_length = input.size(0)
        encoder_outputs = torch.zeros(input_length, self.hidden_size * self.num_directions, device=device) # (seq_len, batch, hidden_size * num_directions)
        encoder_hiddens = []

        encoder_hidden = self.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.forward(
                input[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output
            encoder_hiddens.append(encoder_hidden)

        return encoder_outputs, torch.stack(encoder_hiddens).sum(0)[-1]

    def encode_seq(self, x):
        output = PackedSequence(
            self.embedding(x.data), x.batch_sizes)
        output, hidden = self.rnn(output)
        return output, hidden.squeeze()


if __name__ == '__main__':
    encoder = Encoder()
    print("Encoder")
