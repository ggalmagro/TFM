import numpy as np
from SHADE.SHADE import SHADE
from functions import *
from sklearn.metrics import adjusted_rand_score
import time
from sklearn.externals.joblib import Parallel, delayed
import multiprocessing


def apply(data_set, p_size, nb_clust, max_eval, nb_runs, name, label_percent, labels):

    print("Procesando " + str(name) + " dataset")
    const = np.loadtxt("Restr/" + str(name) + "(" + str(label_percent) + ").txt", dtype=np.int8)
    ml_const, cl_const = get_const_list(const)

    nb_const = len(ml_const) + len(cl_const)
    ml_const_percent = (len(ml_const) / nb_const) * 100
    cl_const_percent = (len(cl_const) / nb_const) * 100

    mean_ars = 0
    mean_execution_time = 0
    mean_unsat_percent = 0

    for j in range(nb_runs):
        shade = SHADE(data_set, ml_const, cl_const, p_size, nb_clust, 0.25, True)
        start = time.clock()
        de_assignment = shade.run(max_eval)[1]
        end = time.clock()
        mean_ars += adjusted_rand_score(labels, de_assignment)
        mean_execution_time += end - start
        mean_unsat_percent += get_usat_percent(ml_const, cl_const, de_assignment)

    mean_ars /= nb_runs
    mean_execution_time /= nb_runs
    mean_unsat_percent /= nb_runs

    return tuple((name, mean_ars, mean_execution_time, mean_unsat_percent))


def main():

    np.random.seed(11)

    #CARGA DE DATOS

    names_array, datasets_array, labels_array = load_all_datasets()

    names_array = names_array[0:5]
    datasets_array = datasets_array[0:5]
    labels_array = labels_array[0:5]

    #BUCLE DE OBTENCION DE DATOS

    const_percent_vector = np.array([0.05, 0.1, 0.15, 0.2])
    #const_percent_vector = np.array([0.1, 0.2])
    nb_runs = 1
    max_eval = 300000
    population_size = 100

    general_start = time.time()
    results_matrix = np.zeros((len(names_array), len(const_percent_vector)))

    with Parallel(n_jobs=4) as parallel:

        for label_percent in range(len(const_percent_vector)):

            print("Porcentaje de restricciones: " + str(const_percent_vector[label_percent]))

            results = parallel(delayed(apply)(datasets_array[i], population_size, len(set(labels_array[i])),
                                              max_eval, nb_runs, names_array[i], const_percent_vector[label_percent],
                                              labels_array[i]) for i in range(len(names_array)))

            for i in range(len(results)):

                results_matrix[i, label_percent] = results[i][1]

            print(results)

    general_end = time.time()

    print("Time --------------> " + str(general_end - general_start) + "\n")

    for i in range(np.shape(results_matrix)[0]):

        print(names_array[i].title(), end='')

        for j in range(np.shape(results_matrix)[1]):

            print(" & %.3f" % (results_matrix[i,j]), end='')

        print(" \\\\")


if __name__ == "__main__": main()