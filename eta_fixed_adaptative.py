from autoencoder import MultilayerPerceptron

from utils.parser import *
from utils.print_letter import is_same_letter
from utils.plots import biplot, biplot_with_new_letter
from config import load_config_multilayer
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

fixed_hidden_layers = [26, 17, 9]
variable_hidden_layers = [28, 22, 17, 10]

f = open('eta_fixed_adaptative.csv', 'w')
writer = csv.writer(f)
writer.writerow(['method', 'wrong_letters', 'time_elapsed'])

for i in range(10):
    letters, labels = load_data_as_bin_array('inputs/font.json')

    # fixed_eta
    autoencoder = MultilayerPerceptron([35] + fixed_hidden_layers + [2] + fixed_hidden_layers[::-1] + [35])
    start = time.process_time()
    autoencoder.train(letters, letters, 100000, 0.0005)
    elapsed = time.process_time() - start

    predictions = np.around(autoencoder.predict(letters), 0)
    wrong_letters, wrong_predictions = is_same_letter(letters, predictions)
    writer.writerow(['fixed_eta', len(wrong_letters), elapsed])
    print("FINISH FIXED " + str(i))
    print("WRONG LETTERS: " + str(len(wrong_letters)))
    print()

    # adaptative_eta
    autoencoder = MultilayerPerceptron([35] + variable_hidden_layers + [2] + variable_hidden_layers[::-1] + [35])
    start = time.process_time()
    autoencoder.train(letters, letters, 10000, adaptative_eta=True)
    elapsed = time.process_time() - start

    predictions = np.around(autoencoder.predict(letters), 0)
    wrong_letters, wrong_predictions = is_same_letter(letters, predictions)
    writer.writerow(['variable_eta', len(wrong_letters), elapsed])
    print("FINISH VARIABLE " + str(i))
    print("WRONG LETTERS: " + str(len(wrong_letters)))
    print()

f.close()