from keras.preprocessing.text import Tokenizer
import json
from .model import chargen_model
from .train_model import *
from .utils import *


class CharGen:
    meta_token = '<s>'
    config = {
        'rnn_layers': 3,
        'rnn_size': 128,
        'input_length': 60,
        'bidirectional': True,
        'embedding_dims': 150
    }
    default_config = config.copy()

    def __init__(self, weights_filepath=None,
                 vocab_filepath=None,
                 config_filepath=None,
                 name='CharGen'):

        if config_filepath is not None:
            with open(config_filepath, 'r',
                      encoding='utf8', errors='ignore') as json_file:
                self.config = json.load(json_file)

        self.config.update({'name': name})
        self.default_config.update({'name': name})
        self.vocab_filepath = vocab_filepath

        if vocab_filepath is not None:
            with open(vocab_filepath, 'r', encoding='utf8', errors='ignore') as json_file:
                self.vocabulary = json.load(json_file)

            self.tokenizer = Tokenizer(filters='', char_level=True)
            self.tokenizer.word_index = self.vocabulary
            self.num_of_classes = len(self.vocabulary) + 1
            self.model = chargen_model(self.num_of_classes,
                                          cfg=self.config,
                                          weights_filepath=weights_filepath)
            self.char_indices = dict((self.vocabulary[c], c) for c in self.vocabulary)



    def generate(self, n=1, temps=[0.3, 0.6, 1.0], gen_text_length=300, prefix=None, **kwargs):
        """Generates as well as returns a single text at each temperature.

        :param n: Number of texts generated at each temperature (default 1)
        :param temps: Float between 0 and 1. Can be a non-list. Higher means more creativity (default [0.3, 0.6, 1.0])
        :param gen_text_length: Max length of the generated text (default 500)
        :param prefix: Uses prefix as the starting seed of the generated text (default None)b n
        :param kwargs:
        :return: None
        """

        for temp in temps:
            print('#'*20 + '\nTemperature: {}\n'.format(temp) +
                  '#'*20)

            text = list(prefix) if prefix else ['']
            gen_text_length += len(text)
            next_character = ''

            if not isinstance(temp, list):
                temp = [temp]

            if model_input_size(self.model) > 1:
                model = Model(inputs=self.model.input[0], outputs=self.model.output[1])

            while next_character != self.meta_token and len(text) < gen_text_length:
                encoded_text = chargen_encode_sequence(text[-self.config['input_length']:],
                                                          self.vocabulary, self.config['input_length'])
                next_temp = temp[(len(text) - 1) % len(temp)]
                next_index = chargen_sample(
                    self.model.predict(encoded_text, batch_size=1)[0],
                    next_temp)
                next_character = self.char_indices[next_index]
                text += [next_character]

            collapse_char = ''

            # if single text, ignore sequences generated w/ padding
            generated_text = collapse_char.join(text)
            generated_text = generated_text[
                          :generated_text.rfind('\n')]  # as text generated in last line might cut-off incorrectly
            print("{}\n".format(generated_text))



    def build_vocab(self, text):
        """Builds the vocabulary needed for training the model and stores as instance variables

        :param text: text to be trained on
        :return: None
        """

        self.tokenizer = Tokenizer(filters='', char_level=True)
        self.tokenizer.fit_on_texts(text)
        self.vocabulary = self.tokenizer.word_index
        self.num_of_classes = len(self.vocabulary) + 1
        self.char_indices = dict((self.vocabulary[c], c) for c in self.vocabulary)

    def save_files(self):
        """Save files for recreating the model

        :return: None
        """
        with open('{}_vocabulary.json'.format(self.config['name']),
                  'w', encoding='utf8') as outfile:
            print(self.tokenizer.word_index)
            json.dump(self.tokenizer.word_index, outfile, ensure_ascii=False)

        with open('{}_config.json'.format(self.config['name']),
                  'w', encoding='utf8') as outfile:
            json.dump(self.config, outfile, ensure_ascii=False)
