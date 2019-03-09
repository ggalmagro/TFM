import numpy as np
from ILSNueva.ILSNueva import ILSNueva
from functions import *
from sklearn.metrics import adjusted_rand_score
import sys


def main():

	np.random.seed(11)

	#CARGA DE DATOS

	names_array, datasets_array, labels_array = load_all_datasets()

	names_array = names_array[0:5]
	datasets_array = datasets_array[0:5]
	labels_array = labels_array[0:5]

	const_percent_vector = [0.05, 0.1, 0.15, 0.2]
	const_array = load_constraints(names_array, const_percent_vector)


	#BUCLE DE OBTENCION DE DATOS

	max_eval = 300000

	for p in range(4, 6):

		xi = -p/10.0

		results_matrix = np.zeros((len(names_array), 4))

		for i in range(len(names_array)):

			print("Procesando " + names_array[i] + " dataset")

			data_set = datasets_array[i]
			labels = labels_array[i]
			nb_clust = len(set(labels))

			for j in range(len(const_array)):

				const = const_array[j][i]
				ml_const, cl_const = get_const_list(const)

				ils = ILSNueva(data_set, ml_const, cl_const, nb_clust, 0.3, 300, 0.3, xi)
				ils_assignment = ils.run(max_eval)[0]
				ars = adjusted_rand_score(labels, ils_assignment)

				results_matrix[i, j] = ars

		print("---------------- Procesando con xi = " + str(xi) + " ----------------")
		for i in range(results_matrix.shape[0]):

			print(names_array[i].title(), end='')

			for j in range(results_matrix.shape[1]):

				print(" & %.3f" % (results_matrix[i, j]), end='')

			print(" \\\\")

if __name__ == "__main__": main()