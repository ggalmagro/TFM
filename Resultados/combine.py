import numpy as np
import csv


def main():

	names_array = ['iris', 'wine', 'soybean', 'pima', 'balance', 'boston', 'diabetes', 
	'breast_cancer', 'bupa', 'ecoli', 'haberman', 'led7digit', 'monk2', 'newthyroid', 
	'vehicle', 'zoo', 'sonar', 'heart', 'ionosphere', 'wdbc', 'vowel', 'movement_libras', 
	'appendicitis', 'saheart', 'spectfheart', 'hayesroth', 'tae', 'glass', 'rand', 'spiral', 'moons', 'circles', "means"]

	data = np.loadtxt("tabla10.dat")
	
	for i in range(data.shape[0]):

		print(names_array[i].title(), end='')

		for j in range(data.shape[1]):

			if j != 3:
				print(" & %.3f" % (data[i, j]), end='')
			else:
				print(" && %.3f" % (data[i, j]), end='')

		print(" \\\\")


if __name__ == "__main__": main()