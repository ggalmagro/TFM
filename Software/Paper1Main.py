import numpy as np
from SHADE.SHADE import SHADE
from BRKGA.BRKGA import BRKGA
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
    results_matrix = np.zeros((len(names_array), 6))

    #BUCLE DE OBTENCION DE DATOS

    nb_runs = 2
    max_eval = 300000
    population_size = 100

    for i in range(len(names_array)):

        print("Procesando " + names_array[i] + " dataset")

        data_set = datasets_array[i]
        labels = labels_array[i]
        nb_clust = len(set(labels))

        const = const_array[0][i]
        ml_const, cl_const = get_const_list(const)

        mean_ars = 0
        mean_execution_time = 0
        mean_unsat_percent = 0

        for j in range(nb_runs):

            shade = SHADE(data_set, ml_const, cl_const, population_size, nb_clust, 0.25, True)
            de_assignment, mean_execution_time = shade.run(max_eval)[1:3]
            mean_ars += adjusted_rand_score(labels, de_assignment)
            mean_unsat_percent += get_usat_percent(ml_const, cl_const, de_assignment)

        mean_ars /= nb_runs
        mean_execution_time /= nb_runs
        mean_unsat_percent /= nb_runs

        results_matrix[i, 0:3] = [mean_ars, mean_unsat_percent, mean_execution_time]

        mean_ars = 0
        mean_execution_time = 0
        mean_unsat_percent = 0

        for j in range(nb_runs):

            brkga = BRKGA(data_set, ml_const, cl_const, population_size, 0.2, 0.2, 0.5, nb_clust, 10)
            brkga_assignment, mean_execution_time = brkga.run(max_eval, True)[1:3]
            mean_ars += adjusted_rand_score(labels, brkga_assignment)
            mean_unsat_percent += get_usat_percent(ml_const, cl_const, brkga_assignment)

        mean_ars /= nb_runs
        mean_execution_time /= nb_runs
        mean_unsat_percent /= nb_runs

        results_matrix[i, 3:6] = [mean_ars, mean_unsat_percent, mean_execution_time]

    for i in range(results_matrix.shape[0]):

        print(names_array[i].title(), end='')

        for j in range(results_matrix.shape[1]):

            if j != 3:
                print(" & %.3f" % (results_matrix[i, j]), end='')
            else:
                print(" && %.3f" % (results_matrix[i, j]), end='')

        print(" \\\\")


if __name__ == "__main__": main()