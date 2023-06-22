import matplotlib.pyplot as plt
import numpy as np


def print_letter(letter):
    monocromatic_cmap = plt.get_cmap('binary')
    plt.imshow(letter)
    plt.show()

def print_noise_letters(original_letters, noise_letters, predictions):
    fig, axes = plt.subplots(3, original_letters.shape[0], figsize=(7,5))
    for i in range(original_letters.shape[0]):
        axes[0, i].set_yticklabels([])
        axes[0, i].set_xticklabels([])
        axes[1, i].set_yticklabels([])
        axes[1, i].set_xticklabels([])
        axes[2, i].set_yticklabels([])
        axes[2, i].set_xticklabels([])
        axes[0, i].imshow(original_letters[i].reshape(7,5))
        axes[1, i].imshow(noise_letters[i].reshape(7,5))
        axes[2, i].imshow(predictions[i].reshape(7,5))
        axes[0, 15].set_title("Original")
        axes[1, 15].set_title("Noise")
        axes[2, 15].set_title("Prediction")
    plt.show()

def print_letters(original_letters, predictions):
    monocromatic_cmap = plt.get_cmap('binary')
    fig, axes = plt.subplots(2, original_letters.shape[0], figsize=(7,5))
    for i in range(original_letters.shape[0]):
        axes[0, i].set_yticklabels([])
        axes[0, i].set_xticklabels([])
        axes[1, i].set_yticklabels([])
        axes[1, i].set_xticklabels([])
        # axes[0, i].axis('off')
        # axes[1, i].axis('off')
        axes[0, i].imshow(original_letters[i].reshape(7,5), cmap=monocromatic_cmap)
        axes[1, i].imshow(predictions[i].reshape(7,5), cmap=monocromatic_cmap)
        axes[0, 15].set_title("Original")
        axes[1, 15].set_title("Prediction")
        
    plt.show()

def is_same_letter(originals: list[float], predictions: list[float], max_errors=1):
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
