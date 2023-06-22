import matplotlib.pyplot as plt
import pandas as pd

def plot_errors_times(file: str):
    df = pd.read_csv(file)

    grouped_by_method = df.groupby('method')

    methods = []
    wrong_letters = []
    wrong_letters_std = []
    time_elapsed = []
    time_elapsed_std = []

    bar_colors = ['tab:red', 'tab:blue']

    for method, grouped_method in grouped_by_method:
        methods.append(method)
        wrong_letters.append(grouped_method['wrong_letters'].mean())
        wrong_letters_std.append(grouped_method['wrong_letters'].std())

        time_elapsed.append(grouped_method['time_elapsed'].mean())
        time_elapsed_std.append(grouped_method['time_elapsed'].std())

    plt.bar(methods, wrong_letters, yerr=wrong_letters_std, color=bar_colors)
    plt.xlabel('Métodos')
    plt.ylabel('Cantidad de símbolos erróneos')
    plt.savefig('figures/' + file + '_errors.png')
    plt.show()

    plt.bar(methods, time_elapsed, yerr=time_elapsed_std, color=bar_colors)
    plt.xlabel('Métodos')
    plt.ylabel('Tiempo de ejecución [s]')
    plt.savefig('figures/' + file + '_times.png')
    plt.show()

plot_errors_times('momentum_adam.csv')
plot_errors_times('eta_fixed_adaptative.csv')