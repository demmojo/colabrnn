from keras.optimizers import RMSprop
from keras.layers import Embedding, Dense, LSTM, Bidirectional
from keras.engine.input_layer import Input
from keras.layers import concatenate, SpatialDropout1D, CuDNNLSTM
from keras.models import Model
from .WeightedAttentionAverage import WeightedAttentionAverage
from keras import backend as K


def chargen_model(num_of_classes, cfg, weights_filepath=None, dropout=0.3, optimizer=RMSprop(lr=4e-3, rho=0.99)):
    """Builds the neural network model architecture for CharGen and loads the weights specified for the model.
    :param num_of_classes: total number of classes
    :param cfg: configuration of CharGen
    :param weights_filepath: path of the weights file
    :param dropout: fraction of neurons to be ignored in a single forward and backward pass (default 0.05)
    :param optimizer: specify which optimization algorithm to use (Default RMSprop)
    :return: model to be used for training
    """

    inp = Input(shape=(cfg['input_length'],), name='input')
    embedded = Embedding(num_of_classes, cfg['embedding_dims'],
                         input_length=cfg['input_length'],
                         name='embedding')(inp)

    if dropout > 0.0:
        embedded = SpatialDropout1D(dropout, name='dropout')(embedded)

    rnn_layer_list = []
    for i in range(cfg['rnn_layers']):
        prev_layer = embedded if i is 0 else rnn_layer_list[-1]
        rnn_layer_list.append(new_rnn_layer(cfg, i + 1)(prev_layer))

    seq_concat = concatenate([embedded] + rnn_layer_list, name='rnn_concat')
    attention = WeightedAttentionAverage(name='attention')(seq_concat)
    output = Dense(num_of_classes, name='output', activation='softmax')(attention)

    model = Model(inputs=[inp], outputs=[output])
    if weights_filepath is not None:
        model.load_weights(weights_filepath, by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def new_rnn_layer(cfg, num_layer):
    """Creates new RNN layer for each parameter depending on whether it is bidirectional LSTM or not.
    :param cfg: configuration of CharGen instance
    :param num_layer: ordinal number of the rnn layer being built
    :return: 3D tensor if return sequence is True
    """
    gpu_no = len(K.tensorflow_backend._get_available_gpus())
    if gpu_no > 0:
        print('GPU is available...')
        if cfg['bidirectional']:
            return Bidirectional(CuDNNLSTM(cfg['rnn_size'],
                                           return_sequences=True),
                                 name='rnn_{}'.format(num_layer))

        return CuDNNLSTM(cfg['rnn_size'],
                         return_sequences=True,
                         name='rnn_{}'.format(num_layer))
    else:
        print('No GPU available...')
        if cfg['bidirectional']:
            return Bidirectional(LSTM(cfg['rnn_size'],
                                      return_sequences=True,
                                      recurrent_activation='sigmoid'),
                                 name='rnn_{}'.format(num_layer))

        return LSTM(cfg['rnn_size'],
                    return_sequences=True,
                    recurrent_activation='sigmoid',
                    name='rnn_{}'.format(num_layer))
