import dynet as dy


class Decoder:
    def __init__(self, param_collection, name):
        self.pc = param_collection.add_subcollection(name)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LSTMDecoder(Decoder):
    def __init__(self, param_collection, nlayers, input_dim, hidden_dim, output_dim, enc_hidden_dim=None,
                 name='LSTMDecoder'):
        """

        :param param_collection: A parameter collection that the decoder will make itself part of.
        :param nlayers: Number of decoder layers.
        :param input_dim: Size of the input vector.
        :param hidden_dim: Size of decoder's hidden output states.
        :param output_dim: Size of target vocabulary.
        :param enc_hidden_dim: Optional size for compatibility with an attention mechanism.
        :param name: Optional name string.
        """
        super().__init__(param_collection, name)
        self.nlayers = nlayers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W_score = self.pc.add_parameters((output_dim, hidden_dim))
        self.b_score = self.pc.add_parameters(output_dim)
        if enc_hidden_dim:
            self.W_attvec = self.pc.add_parameters((hidden_dim, enc_hidden_dim + hidden_dim))
            self.b_attvec = self.pc.add_parameters(hidden_dim)
        self.lstm = dy.LSTMBuilder(self.nlayers, self.input_dim, self.hidden_dim, self.pc)
        self.current_state = None
        self.outputs = None

    def init(self, initial_state, batch_size=1):
        self.current_state = self.lstm.initial_state([initial_state, dy.zeroes((self.hidden_dim,),
                                                                               batch_size=batch_size)])

    def __call__(self, input):
        self.current_state = self.current_state.add_input(input)
        return self.current_state.output()

    def score(self, dec_output, context_vector=None):
        """
        Transform the hidden output of the decoder to output_dim number of scores.
        :param dec_output: Hidden output of the decoder.
        :param context_vector: Optional. Computed by attention mechanism.
        :return: An output_dim score vector that can be normalized into a probability distribution with the softmax.
        """
        if context_vector:  # score the output of the decoder with attention
            att_vector = dy.concatenate([dec_output, context_vector])
            dec_output = dy.tanh(
                dy.affine_transform([dy.parameter(self.b_attvec), dy.parameter(self.W_attvec), att_vector])
            )
        scores = dy.affine_transform([dy.parameter(self.b_score), dy.parameter(self.W_score), dec_output])
        return scores
