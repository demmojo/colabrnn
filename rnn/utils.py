from keras.callbacks import Callback
from keras.models import Model
from keras.preprocessing import sequence
from keras import backend as K
import numpy as np

def chargen_sample(predictions, temp):
    """ Simulates creativity by sampling predicted probabilities of the next character
    """

    predictions = np.asarray(predictions).astype('float64')

    if temp is None or temp == 0.0:
        return np.argmax(predictions)

    predictions = np.log(predictions + K.epsilon()) / temp
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probabilities = np.random.multinomial(1, predictions, 1)
    index = np.argmax(probabilities)

    if index == 0:
        index = np.argsort(predictions)[-2]  # choose second best index from predictions instead of placeholder

    return index

def chargen_encode_sequence(text, vocabulary, maxlen):
    """ Encodes text for prediction with model
    """

    encoded = np.array([vocabulary.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)

def chargen_encode_cat(chars, vocabulary):
    """ One-hot encoding of characters
    """

    arr = np.float32(np.zeros((len(chars), len(vocabulary) + 1)))  # pre-allocates zero matrix
    rows, cols = zip(*[(i, vocabulary.get(char, 0))
                       for i, char in enumerate(chars)])
    arr[rows, cols] = 1
    return arr


def model_input_size(model):
    if isinstance(model.input, list):
        return len(model.input)
    else:   # is a Tensor
        return model.input.shape[0]


class GenerateAfterEpoch(Callback):
    def __init__(self, chargen, gen_epochs, gen_text_length):
        self.chargen = chargen
        self.gen_epochs = gen_epochs
        self.gen_text_length = gen_text_length

    def on_epoch_end(self, epoch, logs={}):
        if self.gen_epochs > 0 and (epoch+1) % self.gen_epochs == 0:
            self.chargen.generate(
                gen_text_length=self.gen_text_length)


class save_model_weights(Callback):
    def __init__(self, weights_name):
        self.weights_name = weights_name

    def on_epoch_end(self, epoch, logs={}):
        if model_input_size(self.model) > 1:
            self.model = Model(inputs=self.model.input[0],
                               outputs=self.model.output[1])
        self.model.save_weights("{}_weights.hdf5".format(self.weights_name))
