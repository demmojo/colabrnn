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

    print(textgen.model.summary())
else:
    textgen = CharGen(name='Test',
                      weights_filepath='/weights/Test_weights.hdf5',
                      vocab_filepath='/vocabulary/Test_vocab.json',
                      config_filepath='config/Test_config.json')

    train('shakespeare.txt', textgen, train_new_model=train_new_model, num_epochs=1)
