from src.Attention import Attention
from src.utils import *


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, encoder_hidden_size=None):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        if not encoder_hidden_size:
            encoder_hidden_size = hidden_size

        self.attention = Attention(hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size + encoder_hidden_size, hidden_size, num_layers=num_layers, dropout=0.5, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, last_hidden, encoder_outputs):
        word_embedded = self.embedding(input)
        attn_weights = self.attention(last_hidden, encoder_outputs)

        context = attn_weights @ encoder_outputs

        # context = (attn_weights * encoder_outputs).sum(0)

        output = torch.cat([word_embedded, context])
        output = output.view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, last_hidden)
        output = self.out(output[0])
        output = self.softmax(output)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def forward_sequence(self, encoder_hidden, encoder_outputs, target_tensor, use_teacher_forcing=True):
        target_length = target_tensor.size(0)
        input_length = encoder_outputs.size(0)
        decoder_outputs = torch.zeros(target_length, self.output_size, device=device)
        decoder_attns = torch.zeros(target_length, input_length, device=device)
        decoder_hidden = encoder_hidden.view(self.num_layers, 1, self.hidden_size)
        decoder_input = torch.tensor(SOS_token, device=device)
        for di in range(target_length):
            decoder_output, decoder_hidden, attn = self.forward(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[di] = decoder_output
            decoder_attns[di] = attn
            if use_teacher_forcing:
                decoder_input = target_tensor[di]  # Teacher forcing
            else:
                decoder_input = decoder_output.argmax().detach()  # detach from history as input
                if decoder_input.item() == EOS_token:
                    break

        return decoder_outputs, decoder_attns

    def decode(self, encoder_hidden, encoder_outputs):
        decoder_outputs = torch.tensor([], device=device)
        decoder_attns = torch.tensor([], device=device)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor(SOS_token, device=device)
        while decoder_input.item() != EOS_token and decoder_outputs.size()[0] < MAX_LENGTH:
            decoder_output, decoder_hidden, attn = self.forward(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs = torch.cat([decoder_outputs, decoder_output])
            decoder_attns = torch.cat([decoder_attns, attn])
        return decoder_outputs, decoder_attns


if __name__ == '__main__':
    print("Decoder")
