from src.utils import *


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def forward_sequence(self, input):
        input_length = input.size(0)
        encoder_outputs = torch.zeros(input_length, self.hidden_size, device=device)
        encoder_hidden = self.initHidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.forward(
                input[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        return encoder_outputs, encoder_hidden


if __name__ == '__main__':
    encoder = Encoder()
    print("Encoder")
