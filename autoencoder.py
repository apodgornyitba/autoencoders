import numpy as np
import copy

class MultilayerPerceptron:
    def __init__(self, layer_sizes, momentum=None, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.momentum = momentum
        self.errors = []

        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        if self.momentum is not None:
            self.prev_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
            self.prev_biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]

        # Adam optimizer variables
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.m_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        self.v_weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        self.m_biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        self.v_biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def feedforward(self, X):
        self.activations = [X]
        self.outputs = []
        for i in range(self.num_layers - 1):
            self.outputs.append(np.dot(self.activations[i], self.weights[i]) + self.biases[i])
            self.activations.append(self.sigmoid(self.outputs[i]))
        return self.activations[-1]
    
    def backpropagation(self, y, learning_rate):
        error = y - self.activations[-1]
        deltas = [error * self.sigmoid_derivative(self.activations[-1])]
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
            deltas.append(delta)
        deltas.reverse()

        # Update weights and biases
        for i in range(self.num_layers - 1):
            grad_weights = np.dot(self.activations[i].T, deltas[i])
            grad_biases = np.sum(deltas[i], axis=0, keepdims=True)

            if self.momentum is not None:
                delta_w = learning_rate * grad_weights + self.momentum * self.prev_weights[i]
                delta_b = learning_rate * grad_biases + self.momentum * self.prev_biases[i]
                self.weights[i] += delta_w
                self.biases[i] += delta_b
                self.prev_weights[i] = copy.deepcopy(delta_w)
                self.prev_biases[i] = copy.deepcopy(delta_b)
            elif self.adam_beta1 is not None and self.adam_beta2 is not None and self.adam_epsilon is not None:
                # Adam optimizer update
                self.m_weights[i] = self.adam_beta1 * self.m_weights[i] + (1 - self.adam_beta1) * grad_weights
                self.v_weights[i] = self.adam_beta2 * self.v_weights[i] + (1 - self.adam_beta2) * np.square(grad_weights)
                m_weights_hat = self.m_weights[i] / (1 - np.power(self.adam_beta1, self.epoch))
                v_weights_hat = self.v_weights[i] / (1 - np.power(self.adam_beta2, self.epoch))
                self.weights[i] += learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.adam_epsilon)

                self.m_biases[i] = self.adam_beta1 * self.m_biases[i] + (1 - self.adam_beta1) * grad_biases
                self.v_biases[i] = self.adam_beta2 * self.v_biases[i] + (1 - self.adam_beta2) * np.square(grad_biases)
                m_biases_hat = self.m_biases[i] / (1 - np.power(self.adam_beta1, self.epoch))
                v_biases_hat = self.v_biases[i] / (1 - np.power(self.adam_beta2, self.epoch))
                self.biases[i] += learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + self.adam_epsilon)
            else:
                self.weights[i] += learning_rate * grad_weights
                self.biases[i] += learning_rate * grad_biases
        
    def train(self, X, y, epochs, learning_rate=0.00005, convergence_threshold=0.01, adaptative_eta=False):
        max_learning_rate = 0.003
        min_learning_rate = 0.00075
        for i in range(epochs):
            self.epoch = i+1
            epoch_learning_rate = learning_rate
            if adaptative_eta:
                epoch_learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * ((epochs - i) / epochs)
                if epoch_learning_rate < min_learning_rate:
                    epoch_learning_rate = min_learning_rate
            self.feedforward(X)
            self.backpropagation(y, epoch_learning_rate)

            self.errors.append(self.mse(y, self.activations[-1]))
            if self.mse(y, self.predict(X)) < convergence_threshold:
                print('Convergence reached at epoch ' + str(i + 1) + '.')
                break
        
    def predict(self, X):
        return self.feedforward(X)

    def mse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def errors(self):
        return self.errors

    def feedforward_to_latent(self, X):
        self.activations = [X]
        self.outputs = []
        for i in range(self.num_layers - 1):
            self.outputs.append(np.dot(self.activations[i], self.weights[i]) + self.biases[i])
            self.activations.append(self.sigmoid(self.outputs[i]))
        return self.activations[int(self.num_layers/2)]

    def latent_predict(self, X):
        self.activations = [X]
        self.outputs = []
        start_layer = int(self.num_layers/2)
        for i in range(start_layer, self.num_layers - 1):
            self.outputs.append(np.dot(self.activations[i - start_layer], self.weights[i]) + self.biases[i])
            self.activations.append(self.sigmoid(self.outputs[i - start_layer]))
        return self.activations[-1]
