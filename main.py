from rnn import CharGen
from rnn.train_model import train

train_new_model = True
model_name = 'shakespeare'

if train_new_model:  # Create a new neural network model and train it
    char_gen = CharGen(name=model_name,
                       bidirectional=True,  # Boolean. Train using a bidirectional LSTM or unidirectional LSTM.
                       rnn_size=128,  # Number of neurons in each layer of your neural network (default 128)
                       rnn_layers=3,  # Number of layers in your neural network (default 3)
                       embedding_dims=100,  # Size of the embedding layer (default 75)
                       input_length=150  # Number of characters considered for prediction (default 60)
                       )
    train(text_filepath='datasets/shakespeare.txt',
          chargen=char_gen,
          gen_text_length=500,  # Number of characters to be generated default 500)
          num_epochs=100,  # Number of times entire dataset is passed forward and backward through the neural network
                           # (default 10)
          batch_size=512,  # Total number of training examples present in a single batch (default 512)
          train_new_model=train_new_model,
          dropout=0.2
          )

    print(char_gen.model.summary())

else:  # Continue training an old model
    text_filename = 'datasets/shakespeare.txt'
    char_gen = CharGen(name=model_name,
                       weights_filepath='models/shakespeare_weights.hdf5',
                       vocab_filepath='models/shakespeare_vocabulary.json',
                       config_filepath='models/shakespeare_config.json')

    train(text_filename, char_gen, train_new_model=train_new_model,
          num_epochs=50)  # change num_epochs to specify number of epochs to continue training

# char_gen = CharGen(weights_filepath='colab_weights.hdf5',  # specify correct filename
#                    vocab_filepath='colab_vocabulary.json',  # specify correct filename
#                    config_filepath='colab_config.json')  # specify correct filename
#
# char_gen.generate(gen_text_length=500, prefix='To be or not to be,')