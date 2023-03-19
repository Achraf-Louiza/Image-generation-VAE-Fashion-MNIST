import tensorflow as tf
from tensorflow import keras

class SamplingLayer(keras.layers.Layer):
    """
    Custom Keras layer that samples from the latent space distribution
    """
    def call(self, inputs):
        """
        Perform the forward pass of the layer.
        
        Args:
        - inputs: a list containing [z_mean, z_log_var], where:
            - z_mean: a tensor of shape (batch_size, latent_dim), representing the mean of the distribution
            - z_log_var: a tensor of shape (batch_size, latent_dim), representing the log-variance of the distribution
        
        Returns:
        - z: a tensor of shape (batch_size, latent_dim), representing a sample from the distribution
        """
        # Unpack the inputs
        z_mean, z_log_var = inputs
        
        # Get the batch size and the number of latent dimensions
        batch_sz = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        
        # Sample epsilon from a normal distribution
        epsilon = tf.random.normal(shape=(batch_sz, latent_dim))
        
        # Compute the sample from the distribution
        z = z_mean + tf.exp(0.5*z_log_var)*epsilon
        
        return z