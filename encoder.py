import dynet as dy


class Encoder:
    def __init__(self, param_collection, name):
        self.pc = param_collection.add_subcollection(name)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class BiLSTMEncoder(Encoder):
    def __init__(self, param_collection, nlayers, input_dim, hidden_dim, name='BiLSTMEncoder'):
        super().__init__(param_collection, name)
        self.nlayers = nlayers
        self.input_dim = input_dim
        assert(hidden_dim % 2 == 0)  # for bidirectionality
        self.hidden_dim = hidden_dim
        self.bilstm = dy.BiRNNBuilder(self.nlayers, self.input_dim, self.hidden_dim, self.pc, dy.LSTMBuilder)
        self.encodings_matrix = None

    def __call__(self, input_sequence):
        encodings = [dy.concatenate([fwd.output(), bwd.output()])
                     for fwd, bwd in self.bilstm.add_inputs(input_sequence)]
        self.encodings_matrix = dy.concatenate_cols(encodings)
        return encodings

    def final_state(self):
        assert(self.encodings_matrix)
        return dy.transpose(self.encodings_matrix)[-1]
