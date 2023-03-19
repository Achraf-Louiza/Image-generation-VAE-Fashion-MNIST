from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, RepeatVector, TimeDistributed, Dropout,
    Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D, Reshape, RepeatVector, BatchNormalization
)
from tensorflow.keras.models import Model
from vae.sampling_layer import SamplingLayer
from vae.variational_loss_layer import VariationalLossLayer

class CVAE:
    def __init__(self, nrows, ncols, latent_dim):
        """
        Constructor for the CVAE (Convolutional Variational Autoencoder) class.

        Arguments:
        sequence_length -- int, length of the input sequences
        n_features -- int, number of features in the input sequences
        latent_dim -- int, dimension of the latent space
        LSTM_cells -- int, number of LSTM cells in the encoder LSTM layer
        """
        self.nrows = nrows
        self.ncols = ncols
        self.latent_dim = latent_dim
        self.create_encoder()
        self.create_decoder()
        self.create_model()

    def create_encoder(self):
        """
        Creates the encoder model of the CVAE.
        """
        # Input: Batch size * sequence length * number of features
        self.encoder_inputs = Input(shape=(self.nrows, self.ncols, 1))

        # Convolution block 1
        x = Conv2D(filters=32, kernel_size=5, padding="same", strides=2, activation="relu")(self.encoder_inputs)
        x = Conv2D(filters=32, kernel_size=3, padding="same", strides=1, activation="relu")(x)
        x = MaxPooling2D()(x)
        
        # Batch normalization
        x = BatchNormalization()(x)
        
        # Dropout
        x = Dropout(0.1)(x)
        
        # Flatten the output of the convolutional layers
        x = Flatten()(x)
        
        # Dense hidden layer
        x = Dense(128)(x)
        
        # Encoder output: Multivariate gaussian distribution per input sequence
        self.z_mean = Dense(units=self.latent_dim, name='Z-Mean')(x)
        self.z_log_sigma = Dense(units=self.latent_dim, name='Z-Log-Sigma')(x)

        # Sampling layer
        self.z = SamplingLayer()([self.z_mean, self.z_log_sigma])

        # Encoder model
        self.encoder = Model(self.encoder_inputs, [self.z_mean, self.z_log_sigma, self.z], name='Encoder-Model')

    def create_decoder(self):
        """
        Creates the decoder model of the CVAE.
        """
        # Fully connected layer
        x = Dense(7*7*32)(self.z)

        # Reshaping for conv2DTranspose
        x = keras.layers.Reshape((7, 7, 32))(x)
        
        # Convolutional transpose block 1
        x = Conv2DTranspose(filters=32, kernel_size=5, padding="same", strides=2, activation="relu")(x)
        x = Conv2DTranspose(filters=32, kernel_size=3, padding="same", strides=2, activation="relu")(x)
        
        # Batch normalization
        x = BatchNormalization()(x)
        
        # Dropout
        x = Dropout(0.1)(x)
        
        # Decoder output: Input reconstruction
        x = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same')(x)
        
        self.decoder = Model(self.z, x, name='Decoder-Model')
        # Decoder final output: A custom variational loss layer with KL & Reconstruction loss functions linked to output layer
        self.decoder_outputs = VariationalLossLayer()([self.encoder_inputs, self.z_mean, self.z_log_sigma, x])
    
    def create_model(self):
        """
        Create the convolution VAE model
        """
        self.model = Model(self.encoder_inputs, self.decoder_outputs, name='CVAE-Model')

    def compile(self, lr=1e-4):
        """
        Compile the model with Adam optimizer and binary cross-entropy loss.

        Args:
            lr (float): Learning rate for the optimizer.
        """
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

    def fit(self, x_train, y_train, x_test, y_test, early_stopper, batch_sz=32, epochs=10):
        """
        Fit the model to the training data.

        Args:
            x_train (ndarray): Training data (input).
            y_train (ndarray): Training data (condition).
            x_test (ndarray): Validation data (input).
            y_test (ndarray): Validation data (condition).
            batch_sz (int): Batch size.
            epochs (int): Number of epochs.
        """
        self.trained = self.model.fit(x_train,
                                      x_train,
                                      epochs=epochs,
                                      batch_size=batch_sz,
                                      validation_data=([x_test, x_test]),
                                      callbacks=[early_stopper],
                                      verbose=True)