from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import Sequence
from keras import backend as K
from .utils import chargen_encode_cat
import numpy as np
from .model import chargen_model
from keras.models import Model, load_model
from keras.preprocessing import sequence
from .utils import *


def train(text_filepath, textgen, num_epochs=50, gen_epochs=1, batch_size=1024, dropout=0.05, train_size=0.8,
          verbose=1, validation=True, gen_text_length=500, train_new_model=True, **kwargs):
    """Trains new model as well as generates samples and saves weights after a specified number of epochs.

    :param text_filepath: the filepath of the text to be trained
    :param textgen: the CharGen instance
    :param num_epochs: number of epochs that model should be trained for (default 50)
    :param gen_epochs: number of epochs after which it generates samples at different temperatures (default 1)
    :param batch_size: number of training examples used in one iteration (default 1024)
    :param dropout: fraction of neurons to be ignored in a single forward and backward pass (default 0.05)
    :param train_size: fraction of the data to be used for training (default .8)
    :param verbose: integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    :param validation: Boolean. Specifies whether or not to conduct validation (default True)
    :param gen_text_length: max length of the generated text (default 500)
    :param train_new_model: Boolean. Specify whether training a new model or not (default True)
    :param kwargs:
    :return: None
    """
    with open(text_filepath, 'r', encoding='utf8', errors='ignore') as f:
        texts = [f.read()]

        print("Training a {}LSTM model with {}-layers each with {} cells".format(
            'Bidirectional ' if textgen.config['bidirectional'] else '',
            textgen.config['rnn_layers'], textgen.config['rnn_size']
        ))

        if train_new_model:
            print('Training a new model...')
            if textgen.vocab_filepath is None:
                textgen.build_vocab(texts)
            textgen.model = chargen_model(textgen.num_of_classes,
                                          dropout=dropout,
                                          cfg=textgen.config)
            textgen.save_files()

        # calculate all of the combinations of token indices and text indices
        list_of_indices = [np.meshgrid(np.array(i), np.arange(
            len(text) + 1)) for i, text in enumerate(texts)]
        list_of_indices = np.block(list_of_indices)

        # Remove the two extra indices
        # Remove initial sequence with padding
        list_of_indices = list_of_indices[textgen.config['input_length']:-2, :]

        indices_mask = np.random.rand(list_of_indices.shape[0]) < train_size

        gen_val = None
        val_steps = None
        if train_size < 1.0 and validation:
            list_of_indices_val = list_of_indices[~indices_mask, :]
            gen_val = generate_sequences_from_texts(
                texts, list_of_indices_val, textgen, batch_size)
            val_steps = max(
                int(np.floor(list_of_indices_val.shape[0] / batch_size)), 1)

        list_of_indices = list_of_indices[indices_mask, :]

        num_tokens = list_of_indices.shape[0]
        assert num_tokens >= batch_size, "Less tokens than the batch_size."

        print("Training on {:,} {} sequences.".format(num_tokens, 'Character'))

        steps_per_epoch = max(int(np.floor(num_tokens / batch_size)), 1)

        gen = generate_sequences_from_texts(
            texts, list_of_indices, textgen, batch_size)

        base_lr = 4e-3

        # inline definition of LearningRateScheduler function
        def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))

        textgen.model.fit_generator(gen, steps_per_epoch=steps_per_epoch,
                                    epochs=num_epochs,
                                    callbacks=[
                                        LearningRateScheduler(lr_linear_decay),
                                        GenerateAfterEpoch(textgen, gen_epochs, gen_text_length), save_model_weights(
                                            textgen.config['name'])],
                                    verbose=verbose,
                                    max_queue_size=2,
                                    validation_data=gen_val,
                                    validation_steps=val_steps
                                    )


def generate_sequences_from_texts(texts, list_of_indices, chargen, batch_size=128):
    input_length = chargen.config['input_length']
    meta_token = chargen.meta_token

    new_tokenizer = chargen.tokenizer

    while True:
        np.random.shuffle(list_of_indices)

        x_batch = []
        y_batch = []
        context_batch = []
        count_batch = 0

        for row in range(list_of_indices.shape[0]):
            text_index = list_of_indices[row, 0]
            end_index = list_of_indices[row, 1]

            text = texts[text_index]

            if end_index > input_length:
                x = text[end_index - input_length: end_index + 1]
            else:
                x = text[0: end_index + 1]
            y = text[end_index + 1]

            if y in chargen.vocabulary:
                x = process_sequence([x], chargen, new_tokenizer)
                y = chargen_encode_cat([y], chargen.vocabulary)

                x_batch.append(x)
                y_batch.append(y)

                count_batch += 1

                if count_batch % batch_size == 0:
                    x_batch = np.squeeze(np.array(x_batch))
                    y_batch = np.squeeze(np.array(y_batch))
                    context_batch = np.squeeze(np.array(context_batch))

                    # print(x_batch.shape)
                    yield (x_batch, y_batch)
                    x_batch = []
                    y_batch = []
                    context_batch = []
                    count_batch = 0


def process_sequence(x, chargen, new_tokenizer):
    x = np.array(x)
    x = new_tokenizer.texts_to_sequences(x)
    x = sequence.pad_sequences(
        x, maxlen=chargen.config['input_length'])

    return x
