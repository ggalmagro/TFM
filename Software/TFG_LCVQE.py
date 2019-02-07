import numpy as np
from LCVQE.LCVQE import LCVQE
from functions import *
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import time
import sys

def get_lcvqe_input(m):
    constraint_number = np.count_nonzero(m - np.identity(np.shape(m)[0])) / 2 + np.shape(m)[0]
    list_const = np.zeros((int(constraint_number), 3), dtype=np.int)
    idx = 0
    for i in range(np.shape(m)[0]):
        for j in range(i + 1, np.shape(m)[0]):
            if m[i, j] != 0:
                list_const[idx, :] = [i, j, m[i, j]]
                idx += 1

    return list_const


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

    nb_runs = 5

    for i in range(len(names_array)):

        print("Procesando " + names_array[i] + " dataset")

        data_set = datasets_array[i]
        labels = labels_array[i]
        nb_clust = len(set(labels))

        const = const_array[0][i]
        centroids = KMeans(init="random", n_clusters=nb_clust).fit(data_set).cluster_centers_
        const_list = get_lcvqe_input(const)
        mean_ars = 0

        for j in range(nb_runs):

            lcvqe_assignment = LCVQE(data_set, nb_clust, const, centroids=centroids)[0]
            mean_ars += adjusted_rand_score(labels, lcvqe_assignment)

        mean_ars /= nb_runs

        results_array.append(mean_ars)

    print("-------------------- Resultados --------------------")

    for i in range(len(results_array)):

        print(names_array[i].title() + " & %.3f" % results_array[i] + " \\\\")


if __name__ == "__main__": main()