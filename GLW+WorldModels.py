
import tensorflow as tf

# Global Latent Workspace modeli
class GlobalLatentWorkspace(tf.keras.Model):
    def __init__(self, latent_dim):
        super(GlobalLatentWorkspace, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(latent_dim) # Latent space dimension
        ])

    def call(self, x):
        encoded = self.encoder(x)
        return encoded

# World Models modeli

class WorldModels(tf.keras.Model):
    def __init__(self, rnn_units, output_dim):
        super(WorldModels, self).__init__()
        self.rnn = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(output_dim)

    def call(self, x, initial_state=None):
        output, state = self.rnn(tf.expand_dims(x, axis=0), initial_state=initial_state)
        output = self.fc(output)
        return output, state

# Global Latent Workspace modelinin oluşturulması
latent_dim = 64
global_latent_model = GlobalLatentWorkspace(latent_dim)

# World Models modelinin oluşturulması
rnn_units = 512
output_dim = 10  # Örnek bir çıkış boyutu
world_model = WorldModels(rnn_units, output_dim)

# Birleştirilmiş modelin oluşturulması
class CombinedModel(tf.keras.Model):
    def __init__(self, global_latent_model, world_model):
        super(CombinedModel, self).__init__()
        self.global_latent_model = global_latent_model
        self.world_model = world_model

    def call(self, x):
        encoded = self.global_latent_model(x)
        output, state = self.world_model(encoded)
        return output, state

# Birleştirilmiş modelin oluşturulması
combined_model = CombinedModel(global_latent_model, world_model)

# Örnek kullanım
# input_data = ... # Giriş verisi, şekil: (batch_size, ...)
# output, state = combined_model(input_data)


import tensorflow as tf
import numpy as np

# Örnek giriş verisi oluşturma
batch_size = 1
input_shape = (28, 28)  # Örnek bir giriş şekli
input_data = np.random.rand(batch_size, *input_shape)  # Örnek giriş verisi

# Örnek giriş verisini TensorFlow tensorüne dönüştürme
input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

# Birleştirilmiş modeli kullanarak çıktı ve iç durumu hesaplama
output, state = combined_model(input_tensor)

# Çıktı ve iç durumu görüntüleme
print("Çıktı şekli:", output.shape)
print("İç durum şekli:", state.shape)
