import numpy as np
from sklearn import datasets
from DE.DE import DE
from functions import *
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import adjusted_rand_score
import time
from sklearn.neighbors.nearest_centroid import NearestCentroid
import scipy.spatial.distance



def get_usat_percent(ml, cl, clustering):

    unsat = 0

    for i in range(len(ml)):

        if clustering[ml[i][0]] != clustering[ml[i][1]]:
            unsat += 1

    for i in range(len(cl)):

        if clustering[cl[i][0]] == clustering[cl[i][1]]:
            unsat += 1

    return (unsat / (len(ml) + len(cl))) * 100


def main():

    np.random.seed(43)

    #CARGA DE DATOS

    names_array, datasets_array, labels_array = load_all_datasets()

    # names_array = [names_array[0]]
    # datasets_array = [datasets_array[0]]
    # labels_array = [labels_array[0]]

    general_table_file = open("Results/DE/general_table_file.txt", "w+")
    results_file = open("Results/DE/results_file.txt", "w+")

    #BUCLE DE OBTENCION DE DATOS

    const_percent_vector = np.array([0.05, 0.1, 0.15, 0.2])
    nb_runs = 1
    max_eval = 100000
    population_size = 120

    for label_percent in const_percent_vector:

        print("------------ Procesando en porcentaje de restricciones: " + str(label_percent) + " ------------")
        general_table_file.write("------------ Procesando en porcentaje de restricciones: " + str(label_percent) + " ------------\n")
        general_table_file.write("Dataset RandIndex   Time(s)   Unsat(%)   TotalRestr   ML(%)   CL(%)\n")
        results_file.write("------------ Procesando en porcentaje de restricciones: " + str(label_percent) + " ------------\n")
        results_file.write("Dataset RandIndex   Time(s)   Unsat(%)   TotalRestr   ML(%)   CL(%)\n")

        for i in range(len(names_array)):

            name = names_array[i]
            data_set = datasets_array[i]
            labels = labels_array[i]
            nb_clust = len(set(labels))

            print("------------ Procesando " + name + " Dataset ------------ (" + str(label_percent) + ")")

            const = np.loadtxt("Restr/" + str(name) + "(" + str(label_percent) + ").txt", dtype=np.int8)
            ml_const, cl_const = get_const_list(const)

            nb_const = len(ml_const) + len(cl_const)
            ml_const_percent = (len(ml_const) / nb_const) * 100
            cl_const_percent = (len(cl_const) / nb_const) * 100

            mean_ars = 0
            mean_execution_time = 0
            mean_unsat_percent = 0

            for j in range(nb_runs):
                de = DE(data_set, ml_const, cl_const, population_size, 1, 0.5, nb_clust, 10)
                start = time.time()
                de_assignment = de.run(max_eval)[1]
                end = time.time()
                mean_ars += adjusted_rand_score(labels, de_assignment)
                mean_execution_time += end - start
                mean_unsat_percent += get_usat_percent(ml_const, cl_const, de_assignment)

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
    # iris_plot1 = draw_data_2DNC(data_set, np.asarray(labels, np.uint8), 3, "DE Iris Dataset ML")
    # iris_plot2 = draw_data_2DNC(data_set, np.asarray(labels, np.uint8), 3, "DE Iris Dataset ML")
    #
    # iris_plot3 = draw_data_2DNC(data_set, np.asarray(de_assignment, np.float), 3,
    #                             "DE Iris Dataset Results ML")
    #
    # iris_plot4 = draw_data_2DNC(data_set, np.asarray(de_assignment, np.float), 3,
    #                             "DE Iris Dataset Results CL")
    #
    # ax1, ax2 = draw_const(data_set, const, iris_plot1, iris_plot2, "Ml Original", "Cl Original")
    #
    # ax3, ax4 = draw_const(data_set, const, iris_plot3, iris_plot4, "Ml Calculado", "Cl Calculado")
    #
    # plt.show()


if __name__ == "__main__": main()