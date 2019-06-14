import numpy as np
from ILSNueva.ILSNueva import ILSNueva
from functions import *
from sklearn.metrics import adjusted_rand_score
import time


def main():

    np.random.seed(43)

    #CARGA DE DATOS (SOLO SE CARGA IRIS)

    names_array, datasets_array, labels_array = load_all_datasets()

    name = names_array[0]
    dataset = datasets_array[0]
    labels = labels_array[0]

    const_array = load_constraints([name], [0.1])
    const = const_array[0][0]

    #BUCLE DE OBTENCION DE DATOS

    nb_runs = 10
    max_gen = 300
    nb_clust = len(set(labels))
    ml_const, cl_const = get_const_list(const)
    xi_array = [-1, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
    xi_array = [-1, -0.5]

    total_restarts_matrix = np.zeros((nb_runs, len(xi_array)))
    mean_restarts_matrix = np.zeros((max_gen, len(xi_array)))

    for xi_pos in range(len(xi_array)):

        print("Procesando con xi = " + str(xi_array[xi_pos]))

        mean_restarts = np.zeros(max_gen)
        total_restarts = []

        for j in range(nb_runs):

            dils = ILSNueva(dataset, ml_const, cl_const, nb_clust, 0.3, max_gen, 0.3, xi_array[xi_pos])
            dils_retarts = dils.run(1000)
            total_restarts.append(dils_retarts[np.size(dils_retarts) -1])
            mean_restarts += dils_retarts

        mean_restarts /= nb_runs
        total_restarts_matrix[:, xi_pos] = total_restarts
        mean_restarts_matrix[:, xi_pos] = mean_restarts


    print("------------------------- total_restarts_matrix -------------------------")
    
    for i in range(np.shape(total_restarts_matrix)[0]):

        print("RowName", end = '')

        for j in range(np.shape(total_restarts_matrix)[1]):

            print(" & %d" % (total_restarts_matrix[i,j]), end = '')

        print(" \\\\")

    print("------------------------- mean_restarts_matrix -------------------------")

    for i in range(np.shape(mean_restarts_matrix)[0]):

        print("%d" % (i + 1), end = '')

        for j in range(np.shape(mean_restarts_matrix)[1]):

            print(", %.2f" % (mean_restarts_matrix[i,j]), end = '')

        print("")



if __name__ == "__main__": main()