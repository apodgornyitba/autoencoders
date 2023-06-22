from autoencoder import MultilayerPerceptron

from utils.parser import *
from utils.print_letter import *
from utils.plots import biplot, biplot_with_new_letter
from config import load_config_multilayer
import numpy as np
import matplotlib.pyplot as plt

def is_same_letter(originals: list[float], predictions: list[float], max_errors=0):
    wrong_letters = []
    wrong_predictions = []
    for i in range(len(originals)):
        errors = 0
        letter = originals[i]
        letter_pred = predictions[i]
        for j in range(len(letter)):
            if letter[j] != int(letter_pred[j]):
                errors += 1
                if errors > max_errors:
                    wrong_letters.append(i)
                    wrong_predictions.append(letter_pred)
                    break
    return wrong_letters, wrong_predictions

hidden_layers = [28, 22, 17, 10]
# hidden_layers = []
# hidden_layers = [30, 25, 20, 15, 10, 6]

autoencoder = MultilayerPerceptron([35] + hidden_layers + [2] + hidden_layers[::-1] + [35], momentum=None)
letters, labels = load_data_as_bin_array('inputs/font.json')
# print(letters)
autoencoder.train(letters, letters, 20000, 0.00005)

# ej1a

latent_predictions = autoencoder.feedforward_to_latent(letters)

# print("a: ", letters[1].reshape(7, 5))
# print("a_pred: ", np.around(autoencoder.predict(letters[1]).reshape(7, 5), 3))
# print('MSE: {}'.format(autoencoder.mse(autoencoder.predict(letters[1]), letters[1])))
# print(autoencoder.predict(letters[1]))

# predictions = np.around(autoencoder.predict(letters), 0)

# wrong_letters, wrong_predictions = is_same_letter(letters, predictions)

# for i in range(len(wrong_letters)):
#     print("Wrong letter: {}".format(labels[wrong_letters[i]]))
#     # print(letters[wrong_letters[i]].reshape(7,5).astype(int))
#     # print(wrong_predictions[i].reshape(7,5).astype(int))
#     # print()

# print_letters(letters, predictions)

# call the function. Use only the 2 PCs.
# biplot(latent_predictions, labels)

# fin ej1a

# ej1d

print("new letters")
x, y = np.random.random(), np.random.random()

print("new coordinates: ({}, {})".format(x, y))


new_letter = autoencoder.latent_predict(np.array([x, y]))

new_letter = np.array(new_letter).reshape(7, 5)

print_letter(new_letter)

biplot_with_new_letter(latent_predictions, labels, x, y)

# if len(wrong_letters) >= 0:
#     for i in range(len(wrong_letters)):
#         print('Wrong letter: {}'.format(labels[wrong_letters[i]]))
#         print('Original:\n{}'.format(letters[wrong_letters[i]].reshape(7, 5)))
#         print('Predicted:\n{}'.format(predictions[i].reshape(7, 5)))
#         print()
# else:
#     print('All letters were guessed correctly')

# print("b prediction")
# print("b: ", autoencoder.predict(letters[2]))
# print('MSE: {}'.format(autoencoder.mse(autoencoder.predict(letters[2]), letters[2])))

# print("c prediction")
# print("c: ", autoencoder.predict(letters[3]))
# print('MSE: {}'.format(autoencoder.mse(autoencoder.predict(letters[3]), letters[3])))

# print("d prediction")
# print("d: ", autoencoder.predict(letters[4]))
# print('MSE: {}'.format(autoencoder.mse(autoencoder.predict(letters[4]), letters[4])))
