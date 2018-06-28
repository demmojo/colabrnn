from rnn import CharGen
from rnn.train_model import train

train_new_model = True

if train_new_model:
    char_gen = CharGen(name="Shakespeare")
    train(text_filepath='shakespeare.txt',
          chargen=char_gen,
          num_epochs=25,
          bidirectional=True,
          rnn_size=128,
          rnn_layers=3,
          batch_size=512,
          embedding_dims=75,
          train_new_model=train_new_model)

    print(char_gen.model.summary())
else:
    chargen = CharGen(name='Shakespeare',
                      weights_filepath='shakespeare_weights.hdf5',
                      vocab_filepath='shakespeare_vocab.json',
                      config_filepath='shakespeare_config.json')

    train('shakespeare.txt', chargen, train_new_model=train_new_model, num_epochs=10)
