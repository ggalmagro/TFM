import numpy as np
import csv


def main():

	names_array = ['iris', 'wine', 'soybean', 'pima', 'balance', 'boston', 'diabetes', 
	'breast_cancer', 'bupa', 'ecoli', 'haberman', 'led7digit', 'monk2', 'newthyroid', 
	'vehicle', 'zoo', 'sonar', 'heart', 'ionosphere', 'wdbc', 'vowel', 'movement_libras', 
	'appendicitis', 'saheart', 'spectfheart', 'hayesroth', 'tae', 'glass', 'rand', 'spiral', 'moons', 'circles', "means"]

	tabla5 = np.loadtxt("ResultadosCompletos/tabla5.dat")
	tabla10 = np.loadtxt("ResultadosCompletos/tabla10.dat")
	tabla15 = np.loadtxt("ResultadosCompletos/tabla15.dat")
	tabla20 = np.loadtxt("ResultadosCompletos/tabla20.dat")

	comp = np.empty((33,4), dtype = np.bool)
	comp2 = np.empty((33,4), dtype = np.float)

	comp[:, 0] = tabla5[:, 0] <= tabla5[:, 3]
	comp[:, 1] = tabla10[:, 0] <= tabla10[:, 3]
	comp[:, 2] = tabla15[:, 0] <= tabla15[:, 3]
	comp[:, 3] = tabla20[:, 0] <= tabla20[:, 3]

	comp2[:, 0] = tabla5[:, 0] - tabla5[:, 3]
	comp2[:, 1] = tabla10[:, 0] - tabla10[:, 3]
	comp2[:, 2] = tabla15[:, 0] - tabla15[:, 3]
	comp2[:, 3] = tabla20[:, 0] - tabla20[:, 3]

	#"Wine ", "Balance ", "Bupa ", "Newthyroid ", "Movement_Libras ", "Heart ", "Rand "

	for i in range(comp.shape[0]):

		print(names_array[i].title(), end='')

		for j in range(comp.shape[1]):

			print(" %r" % (comp[i, j]), end='')


		print("")

	print("--------------------------------------------------------------")

	for i in range(comp2.shape[0]):

		print(names_array[i].title(), end='')

		for j in range(comp2.shape[1]):

			print(" %f" % (comp2[i, j]), end='')


		print("")
	
	


if __name__ == "__main__": main()