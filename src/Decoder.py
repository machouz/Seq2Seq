from src.utils import *


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def forward_sequence(self, encoder_hidden, target_tensor, use_teacher_forcing=True):
        target_length = target_tensor.size(0)
        decoder_outputs = torch.zeros(target_length, self.output_size, device=device)
        decoder_hidden = encoder_hidden

        decoder_input = torch.tensor(SOS_token, device=device)
        for di in range(target_length):
            decoder_output, decoder_hidden = self.forward(
                decoder_input, decoder_hidden)
            decoder_outputs[di] = decoder_output
            if use_teacher_forcing:
                decoder_input = target_tensor[di]  # Teacher forcing
            else:
                decoder_input = decoder_output.argmax().detach()  # detach from history as input
                if decoder_input.item() == EOS_token:
                    break

        return decoder_outputs

    def decode(self,encoder_hidden):
        decoder_outputs = torch.tensor([], device=device)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor(SOS_token, device=device)
        while decoder_input.item() != EOS_token and decoder_outputs.size()[0] < MAX_LENGTH:
            decoder_output, decoder_hidden = self.forward(
                decoder_input, decoder_hidden)
            decoder_outputs = torch.cat([decoder_outputs,decoder_output])
        return decoder_outputs

if __name__ == '__main__':
    print("Decoder")
