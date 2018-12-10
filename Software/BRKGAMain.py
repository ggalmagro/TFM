import numpy as np
from BRKGA.BRKGA import BRKGA
from functions import *
from sklearn.metrics import adjusted_rand_score
import time


def main():

    np.random.seed(43)

    #CARGA DE DATOS

    names_array, datasets_array, labels_array = load_all_datasets()

    names_array = names_array[0:5]
    datasets_array = datasets_array[0:5]
    labels_array = labels_array[0:5]

    const_percent_vector = [0.05, 0.1, 0.15, 0.2]
    #const_percent_vector = [0.1, 0.2]
    const_array = load_constraints(names_array, const_percent_vector)

    general_table_file = open("Results/BRKGA_general_table_file.txt", "w+")
    results_file = open("Results/BRKGA_results_file.txt", "w+")

    #BUCLE DE OBTENCION DE DATOS

    nb_runs = 1
    max_eval = 300000
    population_size = 100
    run_ls = True

    for label_percent in range(len(const_percent_vector)):

        print("------------ Procesando en porcentaje de restricciones: " + str(const_percent_vector[label_percent]) + " ------------")
        general_table_file.write("------------ Procesando en porcentaje de restricciones: " + str(const_percent_vector[label_percent]) + " ------------\n")
        general_table_file.write("Dataset RandIndex   Time(s)   Unsat(%)   TotalRestr   ML(%)   CL(%)\n")
        results_file.write("------------ Procesando en porcentaje de restricciones: " + str(const_percent_vector[label_percent]) + " ------------\n")
        results_file.write("Dataset RandIndex   Time(s)   Unsat(%)\n")

        for i in range(len(names_array)):

            name = names_array[i]
            data_set = datasets_array[i]
            labels = labels_array[i]
            nb_clust = len(set(labels))

            print("------------ Procesando " + name + " Dataset ------------ (" + str(const_percent_vector[label_percent]) + ")")
            const = const_array[label_percent][i]
            ml_const, cl_const = get_const_list(const)

            nb_const = len(ml_const) + len(cl_const)
            ml_const_percent = (len(ml_const) / nb_const) * 100
            cl_const_percent = (len(cl_const) / nb_const) * 100

            mean_ars = 0
            mean_execution_time = 0
            mean_unsat_percent = 0

            for j in range(nb_runs):

                brkga = BRKGA(data_set, ml_const, cl_const, population_size, 0.2, 0.2, 0.5, nb_clust, 10)
                start = time.time()
                brkga_assignment = brkga.run(max_eval, run_ls)[1]
                end = time.time()
                mean_ars += adjusted_rand_score(labels, brkga_assignment)
                mean_execution_time += end - start
                mean_unsat_percent += get_usat_percent(ml_const, cl_const, brkga_assignment)

            mean_ars /= nb_runs
            mean_execution_time /= nb_runs
            mean_unsat_percent /= nb_runs

            print("Restricciones Totales: " + str(nb_const) + " | ML (%): " + str(ml_const_percent) \
                  + " | CL (%): " + str(cl_const_percent))

            print("Rand Index: " + str(mean_ars) + " | Time: " + str(mean_execution_time) + \
                  " | Unsat (%): " + str(mean_unsat_percent))

            general_table_file.write(name + " & " + str(mean_ars) + " & " + str(mean_execution_time) + " & " + str(mean_unsat_percent) + \
                                     " & " + str(nb_const) + " & " + str(ml_const_percent) + " & " + str(cl_const_percent) + " \\\\ \n")

            results_file.write(name + " & " + str(mean_ars) + " & " + str(mean_execution_time) + " & " + str(mean_unsat_percent) + " \\\\ \n")

    general_table_file.close()
    results_file.close()

    #REPRESENTANDO ALGUNOS RESULTADOS
    # iris_plot1 = draw_data_2DNC(iris_set, np.asarray(iris_labels, np.uint8), 3, alg + "Iris Dataset ML")
    # iris_plot2 = draw_data_2DNC(iris_set, np.asarray(iris_labels, np.uint8), 3, alg + "Iris Dataset CL")
    #
    # iris_plot3 = draw_data_2DNC(iris_set, np.asarray(iris_brkga_assignment, np.float), 3, alg + "Iris Dataset Results ML")
    # iris_plot4 = draw_data_2DNC(iris_set, np.asarray(iris_brkga_assignment, np.float), 3, alg + "Iris Dataset Results CL")
    #
    # ax1, ax2 = draw_const(iris_set, iris_const, iris_plot1, iris_plot2, "Ml Original", "Cl Original")
    #
    # ax3, ax4 = draw_const(iris_set, iris_const, iris_plot3, iris_plot4, "Ml Calculado", "Cl Calculado")
    #
    # plt.show()


if __name__ == "__main__": main()