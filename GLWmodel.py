import tensorflow as tf

# Global Latent Workspace Model
class GlobalLatentWorkspace(tf.keras.Model):
    def __init__(self, latent_dim):
        super(GlobalLatentWorkspace, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(latent_dim) # Latent space dimension
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(784, activation='sigmoid'), # Assuming input dimension is 28x28=784
            tf.keras.layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Example usage
latent_dim = 64 # Dimension of the latent space
model = GlobalLatentWorkspace(latent_dim)
