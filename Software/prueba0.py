import numpy as np
from BRKGA.BRKGA import BRKGA
from SHADE.SHADE import SHADE
from functions import *
from sklearn.metrics import adjusted_rand_score
import time


def main():

    np.random.seed(43)

    #CARGA DE DATOS

    names_array, datasets_array, labels_array = load_all_datasets()

    names_array = names_array[0:1]
    datasets_array = datasets_array[0:1]
    labels_array = labels_array[0:1]

    const_percent_vector = [0.05, 0.1, 0.15, 0.2]
    const_percent_vector = [0.1, 0.2]

    const_array = load_constraints(names_array, const_percent_vector)

    file = open("max_eval_comp0.txt", "w+")

    #BUCLE DE OBTENCION DE DATOS

    population_size = 100
    general_start = time.time()

    name = names_array[0]
    data_set = datasets_array[0]
    labels = labels_array[0]

    nb_clust = len(set(labels))

    for i in range(2, 11):

        max_eval = i * 100000

        print("------------ Probando con " + str(max_eval) + " evaluaciones ------------")
        file.write("------------ Probando con " + str(max_eval) + " evaluaciones ------------\n")

        for j in range(len(const_array)):

            for k in range(len(const_array[j])):

                print("------------ Procesando " + name + " Dataset ------------ (" + str(const_percent_vector[j]) + ")")
                const = const_array[j][k]
                ml_const, cl_const = get_const_list(const)

                print("Procesando en porcentaje de restricciones " + str(const_percent_vector[j]))

                file.write(
                    "------------ Procesando en porcentaje de restricciones: " + str(const_percent_vector[j]) + " ------------\n")
                file.write("Dataset RandIndex   Time(s)   Unsat(%)\n")

                print("Procesando con BRKGA")
                brkga = BRKGA(data_set, ml_const, cl_const, population_size, 0.2, 0.2, 0.5, nb_clust, 10)
                start = time.time()
                brkga_assignment = brkga.run(max_eval, True)[1]
                end = time.time()
                brkga_ars = adjusted_rand_score(labels, brkga_assignment)
                brkga_execution_time = end - start
                brkga_unsat_percent = get_usat_percent(ml_const, cl_const, brkga_assignment)

                print("Procesando con SHADE")
                shade = SHADE(data_set, ml_const, cl_const, population_size, nb_clust, True)
                start = time.time()
                de_assignment = shade.run(max_eval)[1]
                end = time.time()
                shade_ars = adjusted_rand_score(labels, de_assignment)
                shade_execution_time = end - start
                shade_unsat_percent = get_usat_percent(ml_const, cl_const, de_assignment)

                file.write("BRKGA -> " + name + " & " + str(brkga_ars) + " & " + str(brkga_execution_time) + " & " +
                           str(brkga_unsat_percent) + "\n")
                file.write("SHADE -> " + name + " & " + str(shade_ars) + " & " + str(shade_execution_time) + " & " +
                           str(shade_unsat_percent) + "\n")

    file.close()
    general_end = time.time()

    print("Time --------------> " + str(general_end - general_start))


if __name__ == "__main__": main()