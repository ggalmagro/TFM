import numpy as np
import csv


def main():
	#"wine ", "balance ", "boston ", "diabetes ", "newthyroid ", "heart ", "rand "
	names_array = ['iris', 'soybean', 'pima', 
	'breast_cancer', 'bupa', 'ecoli', 'haberman', 'led7digit', 'monk2',
	'vehicle', 'zoo', 'sonar', 'ionosphere', 'wdbc', 'vowel', 'movement_libras', 
	'appendicitis', 'saheart', 'spectfheart', 'hayesroth', 'tae', 'glass', 'spiral', 'moons', 'circles', "means"]

	tabla5 = np.loadtxt("tabla5.dat")
	tabla10 = np.loadtxt("tabla10.dat")
	tabla15 = np.loadtxt("tabla15.dat")
	tabla20 = np.loadtxt("tabla20.dat")

	for i in range(len(names_array)):

		print(names_array[i].title(), end='')

		for j in range(tabla5.shape[1]):

			if j != 3:
				print(" & %.3f" % (tabla5[i, j]), end='')
			else:
				print(" && %.3f" % (tabla5[i, j]), end='')

		print(" \\\\")

	print("---------------------------------------------------------------------")

	for i in range(len(names_array)):

		print(names_array[i].title(), end='')

		for j in range(tabla10.shape[1]):

			if j != 3:
				print(" & %.3f" % (tabla10[i, j]), end='')
			else:
				print(" && %.3f" % (tabla10[i, j]), end='')

		print(" \\\\")

	print("---------------------------------------------------------------------")

	for i in range(len(names_array)):

		print(names_array[i].title(), end='')

		for j in range(tabla15.shape[1]):

			if j != 3:
				print(" & %.3f" % (tabla15[i, j]), end='')
			else:
				print(" && %.3f" % (tabla15[i, j]), end='')

		print(" \\\\")

	print("---------------------------------------------------------------------")

	for i in range(len(names_array)):

		print(names_array[i].title(), end='')

		for j in range(tabla20.shape[1]):

			if j != 3:
				print(" & %.3f" % (tabla20[i, j]), end='')
			else:
				print(" && %.3f" % (tabla20[i, j]), end='')

		print(" \\\\")
	
	


if __name__ == "__main__": main()