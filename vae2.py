import numpy as np
import copy

class VariationalAutoencoder:
    def __init__(self, layer_sizes, latent_dim, momentum=None, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.latent_dim = latent_dim
        self.momentum = momentum
        self.errors = []

        self.encoder_weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        self.encoder_biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        
        self.decoder_weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(self.num_layers - 1, 0, -1)]
        self.decoder_biases = [np.zeros((1, self.layer_sizes[i])) for i in range(self.num_layers - 1, 0, -1)]
        
        if self.momentum is not None:
            self.prev_encoder_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
            self.prev_encoder_biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
            
            self.prev_decoder_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1, 0, -1)]
            self.prev_decoder_biases = [np.zeros((1, self.layer_sizes[i])) for i in range(self.num_layers - 1, 0, -1)]

        # Adam optimizer variables
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        
        self.m_encoder_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        self.v_encoder_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        self.m_encoder_biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        self.v_encoder_biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        
        self.m_decoder_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1, 0, -1)]
        self.v_decoder_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1, 0, -1)]
        self.m_decoder_biases = [np.zeros((1, self.layer_sizes[i])) for i in range(self.num_layers - 1, 0, -1)]
        self.v_decoder_biases = [np.zeros((1, self.layer_sizes[i])) for i in range(self.num_layers - 1, 0, -1)]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def reparameterize(self, mean, logvar):
        std = np.exp(0.5 * logvar)
        epsilon = np.random.normal(size=std.shape)
        return mean + std * epsilon
    
    def encoder(self, X):
        self.encoder_activations = [X]
        self.encoder_outputs = []
        for i in range(self.num_layers - 1):
            self.encoder_outputs.append(np.dot(self.encoder_activations[i], self.encoder_weights[i]) + self.encoder_biases[i])
            self.encoder_activations.append(self.sigmoid(self.encoder_outputs[i]))
        mean = self.encoder_activations[-1]
        logvar = self.encoder_outputs[-1]
        return mean, logvar
    
    def decoder(self, latent):
        self.decoder_activations = [latent]
        self.decoder_outputs = []
        for i in range(self.num_layers - 1):
            self.decoder_outputs.append(np.dot(self.decoder_activations[i], self.decoder_weights[i]) + self.decoder_biases[i])
            self.decoder_activations.append(self.sigmoid(self.decoder_outputs[i]))
        return self.decoder_activations[-1]
    
    def feedforward(self, X):
        mean, logvar = self.encoder(X)
        latent = self.reparameterize(mean, logvar)
        return self.decoder(latent)
    
    def backpropagation(self, X, learning_rate):
        mean, logvar = self.encoder(X)
        latent = self.reparameterize(mean, logvar)
        decoded_output = self.decoder(latent)
        
        error = X - decoded_output
        
        # Compute gradients for decoder
        decoder_deltas = [error * self.sigmoid_derivative(self.decoder_activations[-1])]
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(decoder_deltas[-1], self.decoder_weights[i].T) * self.sigmoid_derivative(self.decoder_activations[i])
            decoder_deltas.append(delta)
        decoder_deltas.reverse()

        # Compute gradients for encoder
        encoder_deltas = [np.dot(decoder_deltas[-1], self.decoder_weights[0].T) * self.sigmoid_derivative(self.encoder_activations[-1])]
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(encoder_deltas[-1], self.encoder_weights[i].T) * self.sigmoid_derivative(self.encoder_activations[i])
            encoder_deltas.append(delta)
        encoder_deltas.reverse()
        
        # Update decoder weights and biases
        for i in range(self.num_layers - 1):
            decoder_grad_weights = np.dot(self.decoder_activations[i].T, decoder_deltas[i])
            decoder_grad_biases = np.sum(decoder_deltas[i], axis=0, keepdims=True)
            
            if self.momentum is not None:
                delta_w = learning_rate * decoder_grad_weights + self.momentum * self.prev_decoder_weights[i]
                delta_b = learning_rate * decoder_grad_biases + self.momentum * self.prev_decoder_biases[i]
                self.decoder_weights[i] += delta_w
                self.decoder_biases[i] += delta_b
                self.prev_decoder_weights[i] = copy.deepcopy(delta_w)
                self.prev_decoder_biases[i] = copy.deepcopy(delta_b)
            elif self.adam_beta1 is not None and self.adam_beta2 is not None and self.adam_epsilon is not None:
                # Adam optimizer update for decoder
                self.m_decoder_weights[i] = self.adam_beta1 * self.m_decoder_weights[i] + (1 - self.adam_beta1) * decoder_grad_weights
                self.v_decoder_weights[i] = self.adam_beta2 * self.v_decoder_weights[i] + (1 - self.adam_beta2) * np.square(decoder_grad_weights)
                m_weights_hat = self.m_decoder_weights[i] / (1 - np.power(self.adam_beta1, self.epoch))
                v_weights_hat = self.v_decoder_weights[i] / (1 - np.power(self.adam_beta2, self.epoch))
                self.decoder_weights[i] += learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.adam_epsilon)

                self.m_decoder_biases[i] = self.adam_beta1 * self.m_decoder_biases[i] + (1 - self.adam_beta1) * decoder_grad_biases
                self.v_decoder_biases[i] = self.adam_beta2 * self.v_decoder_biases[i] + (1 - self.adam_beta2) * np.square(decoder_grad_biases)
                m_biases_hat = self.m_decoder_biases[i] / (1 - np.power(self.adam_beta1, self.epoch))
                v_biases_hat = self.v_decoder_biases[i] / (1 - np.power(self.adam_beta2, self.epoch))
                self.decoder_biases[i] += learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + self.adam_epsilon)
            else:
                self.decoder_weights[i] += learning_rate * decoder_grad_weights
                self.decoder_biases[i] += learning_rate * decoder_grad_biases
        
        # Update encoder weights and biases
        for i in range(self.num_layers - 1):
            encoder_grad_weights = np.dot(self.encoder_activations[i].T, encoder_deltas[i])
            encoder_grad_biases = np.sum(encoder_deltas[i], axis=0, keepdims=True)
            
            if self.momentum is not None:
                delta_w = learning_rate * encoder_grad_weights + self.momentum * self.prev_encoder_weights[i]
                delta_b = learning_rate * encoder_grad_biases + self.momentum * self.prev_encoder_biases[i]
                self.encoder_weights[i] += delta_w
                self.encoder_biases[i] += delta_b
                self.prev_encoder_weights[i] = copy.deepcopy(delta_w)
                self.prev_encoder_biases[i] = copy.deepcopy(delta_b)
            elif self.adam_beta1 is not None and self.adam_beta2 is not None and self.adam_epsilon is not None:
                # Adam optimizer update for encoder
                self.m_encoder_weights[i] = self.adam_beta1 * self.m_encoder_weights[i] + (1 - self.adam_beta1) * encoder_grad_weights
                self.v_encoder_weights[i] = self.adam_beta2 * self.v_encoder_weights[i] + (1 - self.adam_beta2) * np.square(encoder_grad_weights)
                m_weights_hat = self.m_encoder_weights[i] / (1 - np.power(self.adam_beta1, self.epoch))
                v_weights_hat = self.v_encoder_weights[i] / (1 - np.power(self.adam_beta2, self.epoch))
                self.encoder_weights[i] += learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.adam_epsilon)

                self.m_encoder_biases[i] = self.adam_beta1 * self.m_encoder_biases[i] + (1 - self.adam_beta1) * encoder_grad_biases
                self.v_encoder_biases[i] = self.adam_beta2 * self.v_encoder_biases[i] + (1 - self.adam_beta2) * np.square(encoder_grad_biases)
                m_biases_hat = self.m_encoder_biases[i] / (1 - np.power(self.adam_beta1, self.epoch))
                v_biases_hat = self.v_encoder_biases[i] / (1 - np.power(self.adam_beta2, self.epoch))
                self.encoder_biases[i] += learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + self.adam_epsilon)
            else:
                self.encoder_weights[i] += learning_rate * encoder_grad_weights
                self.encoder_biases[i] += learning_rate * encoder_grad_biases

        return np.mean(np.abs(error))
    
    def train(self, X, epochs=100, learning_rate=0.001, batch_size=32):
        m = X.shape[0]
        num_batches = m // batch_size
        
        for epoch in range(epochs):
            self.epoch = epoch + 1
            epoch_error = 0.0
            
            for i in range(num_batches):
                batch = X[i*batch_size:(i+1)*batch_size]
                error = self.backpropagation(batch, learning_rate)
                epoch_error += error
            
            self.errors.append(epoch_error / num_batches)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_error / num_batches}")
            
    def generate(self, num_samples):
        latent_samples = np.random.normal(size=(num_samples, self.latent_dim))
        generated_samples = self.decoder(latent_samples)
        return generated_samples
