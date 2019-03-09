import numpy as np
from CEKM.CEKM import CEKM
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

	# datasets_array = datasets_array[5:7]
	# names_array = names_array[5:7]
	# labels_array = labels_array[5:7]
	
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
		broke = False
		for j in range(nb_runs):
			try:
				cekm_assignment = CEKM(data_set, nb_clust, np.asmatrix(const))[0]
				mean_ars += adjusted_rand_score(labels, cekm_assignment)
			except:
				print("Excepcion manejada")
				print(sys.exc_info()[0])
				mean_ars += -1
				broke = True
				break


		if not broke:
			mean_ars /= nb_runs

		results_array.append(mean_ars)
		print(names_array[i].title() + " & %.3f" % results_array[i] + " \\\\")

	print("-------------------- Resultados --------------------")

	for i in range(len(results_array)):

		print(names_array[i].title() + " & %.3f" % results_array[i] + " \\\\")


if __name__ == "__main__": main()