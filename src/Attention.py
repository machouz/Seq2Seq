from src.utils import *


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, decoder_state, encoder_outputs):
        max_len = encoder_outputs.size(0)
        attn_energies = torch.zeros(max_len, self.hidden_size)
        # Calculate energy for each encoder output
        for i in range(max_len):
            attn_energies[i] = self.score(decoder_state.squeeze(), encoder_outputs[i])
        attn_weights = F.softmax(attn_energies, dim=1)

        return attn_weights

    def score(self, decoder_state, encoder_output):
        sj_ci = torch.cat([decoder_state, encoder_output])
        energy = self.attn(sj_ci)
        return energy
