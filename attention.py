import dynet as dy


class Attention:
    def __init__(self, param_collection, name):
        self.pc = param_collection.add_subcollection(name)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class MLPAttention(Attention):  # Bahdanau-style additive
    def __init__(self, param_collection, att_dim, enc_hidden_dim, dec_hidden_dim, name='MLPAttention'):
        super().__init__(param_collection, name)
        self.U_src = self.pc.add_parameters((att_dim, enc_hidden_dim))
        self.W_tgt = self.pc.add_parameters((att_dim, dec_hidden_dim))
        self.V = self.pc.add_parameters(att_dim)
        self.precomputed_src_component = None

    def init(self, src_output_matrix):
        self.precomputed_src_component = dy.parameter(self.U_src) * src_output_matrix

    def __call__(self, src_output_matrix, tgt_output_embedding):
        additive_term = dy.colwise_add(self.precomputed_src_component, dy.parameter(self.W_tgt) * tgt_output_embedding)
        score = dy.transpose(dy.tanh(additive_term)) * dy.parameter(self.V)  # convert to len(src_sent) x 1
        alignment_weights = dy.softmax(score)  # normalize over encoder timesteps
        context_vector = src_output_matrix * alignment_weights
        return context_vector, alignment_weights
