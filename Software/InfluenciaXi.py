import numpy as np
from ILSNueva.ILSNueva import ILSNueva
from functions import *
from sklearn.metrics import adjusted_rand_score
import sys


def main():

	#np.random.seed(11)
    np.random.seed(13)

    #CARGA DE DATOS

    names_array, datasets_array, labels_array = load_all_datasets()

    names_array = names_array[0:1]
    datasets_array = datasets_array[0:1]
    labels_array = labels_array[0:1]

    const_percent_vector = [0.1]
    const_array = load_constraints(names_array, const_percent_vector)
    print(names_array)

    #BUCLE DE OBTENCION DE DATOS

    max_gen = 200
    data_set = datasets_array[0]
    const = const_array[0][0]
    ml_const, cl_const = get_const_list(const)
    results = []

    for p in range(-5, 6):

        xi = p/10.0

        ils = ILSNueva(data_set, ml_const, cl_const, 3, 0.3, 300, 0.3, xi)
        restarts_array = ils.run(max_gen)[2]

        results.append(restarts_array)

    for i in range(max_gen):

        print(i + 1, end='')

        for j in range(len(results)):

            print(" " + str(results[j][i]), end='')
                
        print("")

if __name__ == "__main__": main()