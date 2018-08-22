"""
Implements a sequence-to-sequence model with attention for neural machine translation.
"""
import argparse
import random
import numpy as np
import dynet as dy
from vocab import Vocabulary
from data import read_bitext
from encoder import BiLSTMEncoder
from decoder import LSTMDecoder
from attention import MLPAttention
from batch import prepare_masks
from training import BasicTrainingProcedure
from search import BeamSearch


class Seq2SeqAtt:
    def __init__(self, src_vocab, tgt_vocab, src_emb_dim, tgt_emb_dim,
                 enc_nlayers, enc_hidden_dim,   # encoder settings
                 dec_nlayers, dec_hidden_dim,  # decoder settings
                 att_dim, label_smoothing):
        # Model settings
        self.label_smoothing = label_smoothing
        # Contains all of the model's parameters
        self.pc = dy.ParameterCollection()
        # Vocabulary objects
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        # Embeddings
        self.src_embeddings = self.pc.add_lookup_parameters((len(self.src_vocab), src_emb_dim), name='src-embedding')
        self.tgt_embeddings = self.pc.add_lookup_parameters((len(self.tgt_vocab), tgt_emb_dim), name='tgt-embedding')
        # Init encoder/decoder
        self.encoder = BiLSTMEncoder(self.pc, enc_nlayers, src_emb_dim, enc_hidden_dim)
        self.decoder = LSTMDecoder(self.pc, dec_nlayers, tgt_emb_dim, dec_hidden_dim, len(tgt_vocab), enc_hidden_dim)
        # Init attention
        self.attention = MLPAttention(self.pc, att_dim, enc_hidden_dim, dec_hidden_dim)
        # For affine transform between last state of encoder and first state of decoder
        self.W_bridge = self.pc.add_parameters((dec_hidden_dim, enc_hidden_dim))  # TODO: make class?
        self.b_bridge = self.pc.add_parameters(dec_hidden_dim)
        # Softmax over tgt vocab
        self.W_sm = self.pc.add_parameters((len(tgt_vocab), dec_hidden_dim))
        self.b_sm = self.pc.add_parameters(len(tgt_vocab))
        # For storing matrix of encodings produced by encoder
        self.encodings = None

    def load_model(self, saved_model):
        """
        Load the weights of the saved DyNet model.
        :param saved_model: path to a saved DyNet model
        :return: None
        """
        self.pc.populate(saved_model)

    def cross_entropy_loss(self, scores, next_words):
        if self.label_smoothing:
            log_softmax = dy.log_softmax(scores)
            return -dy.pick_batch(log_softmax, next_words) * (1 - self.label_smoothing) \
                   - dy.mean_elems(log_softmax) * self.label_smoothing
        else:
            return dy.pickneglogsoftmax_batch(scores, next_words)

    def encode(self, src_sents):
        src_sents_by_words = []
        # Make a list of lists of words, where each sublist s_i consists of words in position i
        for i in range(len(max(src_sents, key=len))):
            src_sents_by_words.append([sent[i] for sent in src_sents])
        encodings = self.encoder([dy.lookup_batch(self.src_embeddings, word_col) for word_col in src_sents_by_words])
        self.attention.init(self.encoder.encodings_matrix)  # init attention mechanism with all of the encoder outputs
        return encodings

    def decode(self, prev_words):
        prev_dec_output = self.decoder(dy.lookup_batch(self.tgt_embeddings, prev_words))
        # Using Bahdanau-style attention so we use the previous decoder output
        context_vector, _ = self.attention(self.encoder.encodings_matrix, prev_dec_output)
        scores = self.decoder.score(prev_dec_output, context_vector)
        return scores

    def calc_loss(self, bisents):
        """
        :param bisents: List of (batch size) parallel sentences.
        :return: Average batched loss and number of words processed.
        """
        dy.renew_cg()
        src_sents = [x[0] for x in bisents]
        tgt_sents = [x[1] for x in bisents]
        self.encode(src_sents)
        self.decoder.init(dy.affine_transform([dy.parameter(self.b_bridge), dy.parameter(self.W_bridge),
                                               self.encoder.final_state()]))

        # mask batch
        tgt_sents_by_words, masks, num_words = prepare_masks(tgt_sents, self.tgt_vocab.eos)
        prev_words = tgt_sents_by_words[0]
        all_losses = []

        for next_words, mask in zip(tgt_sents_by_words[1:], masks):
            scores = self.decode(prev_words)  # get the decoder output on the previous time step
            loss = self.cross_entropy_loss(scores, next_words)
            mask_expr = dy.reshape(dy.inputVector(mask), (1,), len(bisents))  # change dimension
            mask_loss = loss * mask_expr
            all_losses.append(mask_loss)
            prev_words = next_words
        return dy.sum_batches(dy.esum(all_losses)), num_words

    def translate(self, bisents, beam_size, max_output_len, length_norm_alpha, output_file,
                  relative, absolute, local, candidate):
        avg_fan_outs = []
        total_fan_outs = []
        with open(output_file, 'w') as output:
            for i in range(len(bisents)):
                print("Translating sentence", i)
                src_sent = bisents[i][0]
                dy.renew_cg()
                self.encode([src_sent])
                self.decoder.init(dy.affine_transform([dy.parameter(self.b_bridge), dy.parameter(self.W_bridge),
                                                       self.encoder.final_state()]))

                beam_search = BeamSearch(beam_size, max_output_len, length_norm_alpha)
                beam_search.set_pruning_strategy(relative, absolute, local, candidate)
                k_best_output, avg_fan_out, total_fan_out, num_pruned = beam_search.search(self)

                print("pruned:", num_pruned)
                print("avg fan out:", avg_fan_out)
                print("total fan out:", total_fan_out)

                # remove start and end symbols
                words = k_best_output[1:-1] if k_best_output[-1] == self.tgt_vocab.eos else k_best_output[1:]
                output_sent = [self.tgt_vocab.i2w[word] for word in words]
                avg_fan_outs.append(avg_fan_out)
                total_fan_outs.append(total_fan_out)
                output.write(" ".join(output_sent) + '\n')
                if (i + 1) % 100 == 0:
                    output.flush()
        print("avg avg fan out:", sum(avg_fan_outs) / len(avg_fan_outs))
        print("avg total fan out:", sum(total_fan_outs) / len(total_fan_outs))

    def save(self, file_name):
        self.pc.save(file_name)


def parse_args():
    """ Parse and return arguments. The default setting is on de -> en. """
    parser = argparse.ArgumentParser()
    # dynet flags to be processed by dynet
    parser.add_argument("--dynet-gpus")
    parser.add_argument("--dynet-mem")
    parser.add_argument("--dynet-autobatch")
    parser.add_argument("--dynet-seed", type=int)
    # model flags
    # data
    parser.add_argument("--train-src", default="../data/bpe_data/train.tok.clean.bpe.40000.de", type=str)
    parser.add_argument("--train-tgt", default="../data/bpe_data/train.tok.clean.bpe.40000.en", type=str)
    parser.add_argument("--dev-src", default="../data/bpe_data/newstest2014.bpe.40000.de", type=str)
    parser.add_argument("--dev-tgt", default="../data/bpe_data/newstest2014.bpe.40000.en", type=str)
    parser.add_argument("--test-src", default="../data/bpe_data/newstest2015.bpe.40000.de", type=str)
    parser.add_argument("--test-tgt", default="../data/bpe_data/newstest2015.bpe.40000.en", type=str)
    # other file paths
    parser.add_argument("--output-file", default="output.en", type=str)
    parser.add_argument("--saved-model", type=str)
    parser.add_argument("--only-decode", action='store_true')
    # hyperparams
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--src-embed-dim", default=300, type=int)
    parser.add_argument("--tgt-embed-dim", default=300, type=int)
    parser.add_argument("--enc-hidden-dim", default=1024, type=int)  # bidirectional
    parser.add_argument("--dec-hidden-dim", default=512, type=int)
    parser.add_argument("--attention-dim", default=256, type=int)
    parser.add_argument("--enc-nlayers", default=1, type=int)
    parser.add_argument("--dec-nlayers", default=1, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--max-output-len", default=80, type=int)
    parser.add_argument("--beam-size", default=1, type=int)
    parser.add_argument("--length-norm", default=None, type=float, help='Alpha for length normalization '
                                                                        'in beam search. Normally, 0.6.')
    parser.add_argument("--relative-pruning", dest="relative", default=None, type=float) # 0.6 in freitag2017
    parser.add_argument("--absolute-pruning", dest="absolute", default=None, type=float) # 2.5 in freitag2017
    parser.add_argument("--local-pruning", dest="local", default=None, type=float)
    parser.add_argument("--max-candidate-pruning", dest="candidate", default=None, type=int)
    parser.add_argument("--vocab-size", default=None, type=int)
    parser.add_argument("--unit", default='lstm', choices=['lstm', 'gru'], type=str,
                        help="Type of gated unit to use for encoder/decoder.")
    parser.add_argument("--label-smoothing", default=None, type=float, help="Epsilon value for label smoothing.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.dynet_seed:
        random.seed(args.dynet_seed)
        np.random.seed(args.dynet_seed)

    src_vocab = Vocabulary('<unk>', eos_symbol='</s>')
    tgt_vocab = Vocabulary('<unk>', sos_symbol='<s>', eos_symbol='</s>')
    train = list(read_bitext(src_vocab, tgt_vocab, args.train_src, args.train_tgt))
    src_vocab.freeze()
    tgt_vocab.freeze()
    dev = list(read_bitext(src_vocab, tgt_vocab, args.dev_src, args.dev_tgt))
    # init model
    model = Seq2SeqAtt(src_vocab, tgt_vocab, args.src_embed_dim, args.tgt_embed_dim,
                       args.enc_nlayers, args.enc_hidden_dim,
                       args.dec_nlayers, args.dec_hidden_dim,
                       args.attention_dim, args.label_smoothing)
    if args.saved_model:
        model.load_model(args.saved_model)
    if args.only_decode:
        print("Reading test data...")
        test = list(read_bitext(src_vocab, tgt_vocab, args.test_src, args.test_tgt))
        model.translate(test, args.beam_size, args.max_output_len, args.length_norm, args.output_file,
                        args.relative, args.absolute, args.local, args.candidate)
        print("Done")
    else:
        training_procedure = BasicTrainingProcedure(model, dy.SimpleSGDTrainer(model.pc))
        training_procedure.train(args.epochs, train, dev, args.batch_size, args.batch_size, args.max_output_len)


if __name__ == '__main__':
    main()

