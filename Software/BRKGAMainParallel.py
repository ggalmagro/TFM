import numpy as np
from BRKGA.BRKGA import BRKGA
from functions import *
from sklearn.metrics import adjusted_rand_score
import time
from sklearn.externals.joblib import Parallel, delayed
import multiprocessing


def apply(data_set, p_size, nb_clust, max_eval, nb_runs, name, const, labels):

    print("Procesando " + str(name) + " dataset")
    ml_const, cl_const = get_const_list(const)

    nb_const = len(ml_const) + len(cl_const)
    ml_const_percent = (len(ml_const) / nb_const) * 100
    cl_const_percent = (len(cl_const) / nb_const) * 100

    mean_ars = 0
    mean_execution_time = 0
    mean_unsat_percent = 0

    for j in range(nb_runs):
        de = BRKGA(data_set, ml_const, cl_const, p_size, 0.2, 0.2, 0.6, nb_clust, 10)
        start = time.time()
        de_assignment = de.run(max_eval)[1]
        end = time.time()
        mean_ars += adjusted_rand_score(labels, de_assignment)
        mean_execution_time += end - start
        mean_unsat_percent += get_usat_percent(ml_const, cl_const, de_assignment)

    mean_ars /= nb_runs
    mean_execution_time /= nb_runs
    mean_unsat_percent /= nb_runs

    return tuple((name, nb_const, ml_const_percent, cl_const_percent, mean_ars, mean_execution_time, mean_unsat_percent))


def main():

    np.random.seed(43)

    #CARGA DE DATOS

    names_array, datasets_array, labels_array = load_all_datasets()

    names_array = names_array[:5]
    datasets_array = datasets_array[:5]
    labels_array = labels_array[:5]

    const_percent_vector = [0.05, 0.1, 0.15, 0.2]
    # const_percent_vector = [0.05]
    const_array = load_constraints(names_array, const_percent_vector)

    general_table_file = open("Results/BRKGA_general_table_file.txt", "w+")
    results_file = open("Results/BRKGA_results_file.txt", "w+")

    #BUCLE DE OBTENCION DE DATOS

    nb_runs = 1
    max_eval = 300000
    population_size = 100

    general_start = time.time()

    with Parallel(n_jobs=multiprocessing.cpu_count() - 1) as parallel:

        const_array_index = 0

        for label_percent in const_percent_vector:

            general_table_file.write(
                "------------ Procesando en porcentaje de restricciones: " + str(label_percent) + " ------------\n")
            general_table_file.write("Dataset RandIndex   Time(s)   Unsat(%)   TotalRestr   ML(%)   CL(%)\n")
            results_file.write(
                "------------ Procesando en porcentaje de restricciones: " + str(label_percent) + " ------------\n")
            results_file.write("Dataset RandIndex   Time(s)   Unsat(%)\n")

            results = parallel(delayed(apply)(datasets_array[i], population_size, len(set(labels_array[i])),
                                              max_eval, nb_runs, names_array[i], const_array[const_array_index][i],
                                              labels_array[i]) for i in range(len(names_array)))

            for i in range(len(results)):
                general_table_file.write(
                    results[i][0] + " & " + str(results[i][4]) + " & " + str(results[i][5]) + " & " + str(results[i][6]) + \
                    " & " + str(results[i][1]) + " & " + str(results[i][2]) + " & " + str(results[i][3]) + " \\\\ \n")

                results_file.write(results[i][0] + " & " + str(results[i][4]) + " & " + str(results[i][5]) + " & " + str(
                    results[i][6]) + " \\\\ \n")

            print(results)
            const_array_index += 1

    general_table_file.close()
    results_file.close()

    general_end = time.time()

    print("Time --------------> " + str(general_end - general_start))


if __name__ == "__main__": main()