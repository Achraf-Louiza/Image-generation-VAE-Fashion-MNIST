import tensorflow as tf
from tensorflow import keras
    
class VariationalLossLayer(keras.layers.Layer):
    """
    Custom Keras layer that computes the VAE loss function
    """
    def __init__(self, loss_weights=[1, 1e-3]):
        """
        Constructor method for the layer.
        
        Args:
        - loss_weights: a list containing the weights to apply to the reconstruction loss and the KL divergence loss,
        respectively
        """
        super().__init__()
        self.k1 = loss_weights[0]
        self.k2 = loss_weights[1]
        
    def call(self, inputs):
        """
        Perform the forward pass of the layer.
        
        Args:
        - inputs: a list containing [x, z_mean, z_log_var, y], where:
            - x: a tensor of shape (batch_size, sequence_length, n_features), representing the input data
            - z_mean: a tensor of shape (batch_size, latent_dim), representing the mean of the distribution
            - z_log_var: a tensor of shape (batch_size, latent_dim), representing the log-variance of the distribution
            - y: a tensor of shape (batch_size, sequence_length, n_features), representing the reconstructed data
        
        Returns:
        - y: a tensor of shape (batch_size, sequence_length, n_features), representing the reconstructed data
        """
        # Unpack the inputs
        x, z_mean, z_log_var, y = inputs
        
        # Compute the reconstruction loss
        r_loss = self.k1 * keras.losses.binary_crossentropy(x, y)
        
        # Compute the KL divergence loss
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = self.k2 * (-0.5*tf.reduce_mean(kl_loss))
        
        # Compute the total loss
        loss = r_loss + kl_loss
        
        # Add the loss to the layer
        self.add_loss(loss)
        
        # Add the loss and its components as metrics
        self.add_metric(loss, aggregation='mean', name='loss')
        self.add_metric(r_loss/self.k1, aggregation='mean', name='r_loss')
        self.add_metric(kl_loss/self.k2, aggregation='mean', name='kl_loss')
        
        return y
    
    def get_config(self):
        """
        Get the configuration of the layer.
        """
        return {'loss_weights': [self.k1, self.k2]}

