import numpy as np

# add noise to a flatten matrix
def add_toggled_noise(letters, noise):
    all_pixels = range(len(letters[0]))
    pixel_amount = len(all_pixels)
    for i in range(len(letters)):
        noised_pixels = np.random.choice(all_pixels, int(np.round(noise * pixel_amount, 0)))
        for pixel in noised_pixels:
            letters[i][pixel] = 1 - letters[i][pixel]
        
    return letters

# add noise to a flatten matrix
def add_zeroed_noise(letters, noise):
    all_pixels = range(len(letters[0]))
    pixel_amount = len(all_pixels)
    for i in range(len(letters)):
        noised_pixels = np.random.choice(all_pixels, int(np.round(noise * pixel_amount, 0)))
        for pixel in noised_pixels:
            letters[i][pixel] = 0
        
    return letters