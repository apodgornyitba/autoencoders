import numpy as np
import json

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    labels=['`', 'a', 'b', 'c' ,'d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n' ,'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y' ,'z', '{',  '|', '}', '~', 'DEL']
    return data['font'], labels

def to_bin_array(encoded_caracter):
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(0, 7):
        current_row = encoded_caracter[row]
        for col in range(0, 5):
            bin_array[row][4-col] = current_row & 1
            current_row >>= 1
    return bin_array.flatten()

def load_data_as_bin_array(path):
    letters, labels = load_data(path)
    bin_letters = []
    for letter in letters:
        bin_letters.append(to_bin_array(letter))
    return np.array(bin_letters), labels

