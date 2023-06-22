from autoencoder import MultilayerPerceptron

from utils.parser import *
from utils.print_letter import is_same_letter
from utils.plots import biplot, biplot_with_new_letter
from config import load_config_multilayer
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

momentum_hidden_layers = [22, 10]
adam_hidden_layers = [26, 17, 9]

f = open('momentum_adam.csv', 'w')
writer = csv.writer(f)
writer.writerow(['method', 'wrong_letters', 'time_elapsed'])

letters, labels = load_data_as_bin_array('inputs/font.json')

for i in range(10):
    # momentum
    autoencoder = MultilayerPerceptron([35] + momentum_hidden_layers + [2] + momentum_hidden_layers[::-1] + [35], momentum=0.9)
    start = time.process_time()
    autoencoder.train(letters, letters, 20000, 0.005)
    elapsed = time.process_time() - start

    predictions = np.around(autoencoder.predict(letters), 0)
    wrong_letters, wrong_predictions = is_same_letter(letters, predictions)
    writer.writerow(['momentum', len(wrong_letters), elapsed])
    print("FINISH MOMENTUM " + str(i))
    print("WRONG LETTERS: " + str(len(wrong_letters)))
    print()

    # adam
    autoencoder = MultilayerPerceptron([35] + adam_hidden_layers + [2] + adam_hidden_layers[::-1] + [35])
    start = time.process_time()
    autoencoder.train(letters, letters, 60000, 0.0005)
    elapsed = time.process_time() - start

    predictions = np.around(autoencoder.predict(letters), 0)
    wrong_letters, wrong_predictions = is_same_letter(letters, predictions)
    writer.writerow(['adam', len(wrong_letters), elapsed])
    print("FINISH ADAM " + str(i))
    print("WRONG LETTERS: " + str(len(wrong_letters)))
    print()

f.close()