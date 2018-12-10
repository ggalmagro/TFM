from functions import *
import numpy as np
import collections

def main():

	names_array, datasets_array, labels_array = load_all_datasets()
	#names_array = names_array[0:4]

	for i in range(len(names_array)):

		n = np.shape(datasets_array[i])[0]
		nb_class = len(set(labels_array[i]))

		print(str(n) + " " + names_array[i].title() + " & " + str(n) + " & " + str(nb_class) + " &  \\\\")


	const_percent_vector = [0.05, 0.1, 0.15, 0.2]
	const_array = load_constraints(names_array, const_percent_vector)

	for i in range(len(names_array)):

		print(names_array[i].title(), end='')

		for l in range(len(const_percent_vector)):

			c = const_array[l][i]
			total = np.count_nonzero(c)
			total = (total - np.shape(c)[0]) / 2
			unique, counts = np.unique(c, return_counts=True)
			d = dict(zip(unique, counts))
			ml = (d[1] - np.shape(c)[0]) / 2
			cl = d[-1] / 2

			print(" & %d & %d & %d" % (total, ml, cl), end='')

		print(" \\\\")




if __name__ == "__main__": main()