import numpy as np
from COPKmeans.COPKmeans import COPKM
from functions import *
from sklearn.metrics import adjusted_rand_score
import time
import sys


def main():

    if len(sys.argv) != 2:
        print("Numero de argumentos incorrecto")
        return -1

    np.random.seed(11)

    #CARGA DE DATOS

    names_array, datasets_array, labels_array = load_all_datasets()
    
    const_percent_vector = [float(sys.argv[1])]
    const_array = load_constraints(names_array, const_percent_vector)
    results_array = []

    #BUCLE DE OBTENCION DE DATOS

    nb_runs = 1

    for i in range(len(names_array)):

        print("Procesando " + names_array[i] + " dataset")

        data_set = datasets_array[i]
        labels = labels_array[i]
        nb_clust = len(set(labels))

        const = const_array[0][i]

        mean_ars = 0

        for j in range(nb_runs):

            copkm_assignment, wine_centroid = COPKM(data_set, nb_clust, const, max_iter = 300)

            if copkm_assignment == None:
            	mean_ars = -1
            else:
            	mean_ars += adjusted_rand_score(labels, copkm_assignment)

        mean_ars /= nb_runs

        results_array.append(mean_ars)

    print("-------------------- Resultados --------------------")

    for i in range(len(results_array)):

        print(names_array[i].title() + " & %.3f" % results_array[i] + " \\\\")


if __name__ == "__main__": main()