from src.utils import *
from src.Encoder import Encoder
from src.Decoder import Decoder

teacher_forcing_ratio = 0.5
data_cache_fname = "prepared_data"


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          teacher_ratio=False):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder.forward_sequence(input_tensor)
    decoder_outputs = decoder.forward_sequence(encoder_hidden, target_tensor, teacher_ratio)

    loss = criterion(decoder_outputs.unsqueeze(-1), target_tensor)
    correct = (decoder_outputs.argmax(1) == target_tensor.squeeze()).sum()
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / decoder_outputs.size(0), correct.item() / decoder_outputs.size(0)


def trainIters(encoder, decoder, training_pairs, n_iters, print_every=100, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    print_accuracy_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        if (iter % 5 == 0):
            loss, accuracy = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                   criterion, False)
        else:
            loss, accuracy = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                   criterion, True)
        print_loss_total += loss
        print_accuracy_total += accuracy

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_accuracy_avg = print_accuracy_total / print_every
            print_loss_total = 0
            print_accuracy_total = 0
            print('%s (%d %d%%) loss: %.4f accuracy: %.4f,  ' % (timeSince(start, iter / n_iters),
                                                                 iter, iter / n_iters * 100, print_loss_avg,
                                                                 print_accuracy_avg))


def translate(input, input_lang, output_lang, encoder, decoder):
    input_tensor = tensorFromSentence(input_lang, input)
    encoder_outputs, encoder_hidden = encoder.forward_sequence(input_tensor)
    decoder_outputs = decoder.decode(encoder_hidden)
    return input, sentenceFromTensor(output_lang, decoder_outputs.argmax(1))


if __name__ == '__main__':
    prepared_data = loadElement(data_cache_fname)

    if not prepared_data:
        print("Loading data from file")
        prepared_data = prepareData('eng', 'fra', True)
        saveElement(prepared_data, data_cache_fname)
    else:
        print("Loaded data from cache")
    input_lang, output_lang, pairs = prepared_data
    p = random.choice(pairs)
    t = tensorsFromPair(input_lang, output_lang, p)
    t = pairFromTensor(input_lang, output_lang, t)
    encoder = Encoder(len(input_lang.word2index) + 2, 50)
    decoder = Decoder(50, len(output_lang.word2index) + 1)
    criterion = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

    n_iters = 5000
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]
    trainIters(encoder, decoder, training_pairs, n_iters=n_iters)
