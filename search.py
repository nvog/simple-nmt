import dynet as dy
from collections import namedtuple, defaultdict
import numpy as np
from scipy.special import logsumexp


class BeamSearch:
    Beam = namedtuple('Beam', ['state', 'words', 'log_prob', 'last_word_prob', 'len_norm_score', 'candidate_id'])
    PruningStrategy = namedtuple('PruningStrategy', ['relative', 'absolute', 'local', 'candidate'])

    def __init__(self, beam_size, max_output_len, length_norm_alpha):
        self.beam_size = beam_size
        self.max_output_len = max_output_len
        self.pruning_strategy = None
        self.length_norm_alpha = length_norm_alpha

    def set_pruning_strategy(self, relative=None, absolute=None, local=None, candidate=None):
        self.pruning_strategy = self.PruningStrategy(np.log(relative) if relative else None,
                                                     np.log(absolute) if absolute else None,
                                                     np.log(local) if local else None,
                                                     candidate)

    def search(self, translation_model):
        beams = [self.Beam(translation_model.decoder.current_state, [translation_model.tgt_vocab.sos],
                           0.0, 0.0, 0.0, 1)]
        next_candidate_id = 2
        num_pruned = 0
        fan_outs = []
        target_vocabulary_size = len(translation_model.tgt_vocab)
        for i in range(self.max_output_len):
            probabilities = []
            next_states = []
            for beam in beams:
                # if already at end of sentence, no work to be done
                if beam.words[-1] == translation_model.tgt_vocab.eos:
                    probabilities.append(dy.zeroes((target_vocabulary_size,)) + 1)
                    next_states.append(None)
                    continue

                # calculate decoding scores
                scores = translation_model.decode([beam.words[-1]], beam.state)
                # then, keep track of next decoder state
                next_states.append(translation_model.decoder.current_state)
                probabilities.append(dy.log_softmax(scores))

            # run forward pass
            probabilities = dy.concatenate_to_batch(probabilities).npvalue().T.reshape(-1, target_vocabulary_size)

            new_beams = []
            for prob, beam, next_state in zip(probabilities, beams, next_states):
                if beam.words[-1] == translation_model.tgt_vocab.eos:
                    # if we're already at the end of the sentence, keep it as is
                    new_beams.append(beam)
                else:
                    # otherwise, find the k best candidate words
                    k_best = np.argsort(prob)  # best is last
                    for next_word in k_best[-self.beam_size:]:
                        next_word_prob = prob[next_word]
                        new_prob = beam.log_prob + next_word_prob
                        if self.length_norm_alpha:
                            len_norm = (5 + len(beam.words) + 1)**self.length_norm_alpha / (5 + 1) ** self.length_norm_alpha
                        else:
                            len_norm = 1
                        new_beams.append(
                            self.Beam(next_state, beam.words + [next_word], new_prob, next_word_prob,
                                      new_prob / len_norm, next_candidate_id))
                    next_candidate_id += 1

            # Only keep the k best
            beams = sorted(new_beams, key=lambda beam: beam.len_norm_score)[-self.beam_size:]
            # if highest scoring candidate is a complete sentence, exit
            if beams[-1].words[-1] == translation_model.tgt_vocab.eos:
                break

            best_score = beams[-1].len_norm_score
            if self.pruning_strategy.relative:
                beams = [beam for beam in beams if beam.len_norm_score - best_score > self.pruning_strategy.relative]
            if self.pruning_strategy.absolute:
                beams = [beam for beam in beams if
                         logsumexp(a=[best_score, beam.len_norm_score], b=[1, -1]) < self.pruning_strategy.absolute]
            if self.pruning_strategy.local:
                best_word_score = max(beam.last_word_prob for beam in beams)
                beams = [beam for beam in beams if beam.last_word_prob - best_word_score > self.pruning_strategy.local]
            if self.pruning_strategy.candidate:
                pruned_beams = []
                candidate_counts = defaultdict(lambda: 0)
                for beam in reversed(beams):
                    if candidate_counts[beam.candidate_id] < self.pruning_strategy.candidate:
                        pruned_beams.insert(0, beam)
                        candidate_counts[beam.candidate_id] += 1
                beams = pruned_beams
            num_pruned += self.beam_size - len(beams)
            fan_out = 0
            for beam in beams:
                if beam.words[-1] != translation_model.tgt_vocab.eos:
                    fan_out += 1
            fan_outs.append(fan_out)

        total_fan_out = sum(fan_outs)
        avg_fan_out = total_fan_out / len(fan_outs) if len(fan_outs) != 0 else 0
        return beams[-1].words, avg_fan_out, total_fan_out, num_pruned
