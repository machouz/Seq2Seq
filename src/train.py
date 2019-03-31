from src.AttentionDecoder import AttentionDecoder
from src.utils import *
from src.Encoder import Encoder
from src.Decoder import Decoder

teacher_forcing_ratio = 0.5
data_cache_fname = "{}_prepared_data".format(device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          teacher_ratio=False):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder.forward_sequence(input_tensor)
    decoder_outputs, decoder_attns = decoder.forward_sequence(encoder_hidden, encoder_outputs, target_tensor,
                                                              teacher_ratio)

    loss = criterion(decoder_outputs, target_tensor)
    correct = (decoder_outputs.argmax(1) == target_tensor.squeeze()).sum()
    loss.backward()
    if ((correct.item() - 2) / (target_tensor.size(0) - 2) > 0.8):
        print('Original : %s' % sentenceFromTensor(input_lang, input_tensor))
        print('Translation : %s' % sentenceFromTensor(output_lang, target_tensor))
        print('Prediction : %s' % sentenceFromTensor(output_lang, decoder_outputs.argmax(1)))
        print('Attention : %s' % decoder_attns)
        fig = show_attention(sentenceFromTensor(input_lang, input_tensor),
                             sentenceFromTensor(output_lang, decoder_outputs.argmax(1)), decoder_attns.detach())
        plt.close(fig)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / decoder_outputs.size(0), correct.item() / decoder_outputs.size(0)


def trainIters(encoder, decoder, training_pairs, epochs, print_every=100, learning_rate=0.001):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    print_accuracy_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    n_iters = len(training_pairs)
    for epoch in range(1, epochs + 1):
        for iter, (input_tensor, target_tensor) in enumerate(training_pairs):
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            loss, accuracy = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                                       decoder_optimizer,
                                       criterion, use_teacher_forcing)
            print_loss_total += loss
            print_accuracy_total += accuracy

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_accuracy_avg = print_accuracy_total / print_every
                print_loss_total = 0
                print_accuracy_total = 0
                print('%s (%d %d%%) loss: %.4f accuracy: %.4f,  ' % (timeSince(start, iter + 1 / n_iters),
                                                                     iter, iter / n_iters * 100, print_loss_avg,
                                                                     print_accuracy_avg))


def translate(input, input_lang, output_lang, encoder, decoder):
    encoder.eval()
    decoder.eval()
    input_tensor = tensorFromSentence(input_lang, input)
    encoder_outputs, encoder_hidden = encoder.forward_sequence(input_tensor)
    decoder_outputs, decoder_attns = decoder.decode(encoder_hidden,encoder_outputs)

    encoder.train()
    decoder.train()
    print('Original : %s' % input)
    print('Translation : %s' % sentenceFromTensor(output_lang, decoder_outputs.argmax(1)))
    return decoder_outputs, decoder_attns


if __name__ == '__main__':
    prepared_data = loadElement(data_cache_fname)

    if not prepared_data:
        print("Loading data from file")
        prepared_data = prepareData('eng', 'fra', True)
        saveElement(prepared_data, data_cache_fname)
    else:
        print("Loaded data from cache")
    input_lang, output_lang, pairs = prepared_data
    random.shuffle(pairs)
    p = random.choice(pairs)
    t = tensorsFromPair(input_lang, output_lang, p)
    t = pairFromTensor(input_lang, output_lang, t)
    encoder = Encoder(len(input_lang.word2index) + 2, 50)
    encoder.to(device)
    decoder = AttentionDecoder(50, len(output_lang.word2index) + 2)
    decoder.to(device)

    training_pairs = [tensorsFromPair(input_lang, output_lang, p)
                      for p in pairs]
    translate("vous etes bon .", input_lang, output_lang, encoder, decoder)
    trainIters(encoder, decoder, training_pairs, epochs=1)
